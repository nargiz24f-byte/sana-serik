import express from "express";
import dotenv from "dotenv";
import path from "path";
import fs from "fs/promises";
import crypto from "crypto";
import { fileURLToPath } from "url";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 3000);
const MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const TEACHER_PASSWORD = process.env.TEACHER_PASSWORD || "change-this-password";
const apiKey = process.env.GEMINI_API_KEY;
const groqApiKey = process.env.GROQ_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_FILE = path.join(__dirname, "data", "submissions.json");

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

const SYSTEM_INSTRUCTION = `
Сен — «САНА-СЕРІК», қазақ тілі сабағында эссе жазуға көмектесетін ЖИ-ойлау серігісің.
Мақсат — оқушының орнына жазу емес, оның өз ойын дамыту.

Міндетті қағидалар:
- Дайын эссе, толық дайын абзац немесе көшіріп тапсыратын мәтін берме.
- Оқушы дайын мәтін сұраса, сыпайы бас тартып, жетекші сұрақтар, жоспар немесе идеяны дамыту қадамдарын бер.
- Оқушының өз позициясын, аргументін, дәлелін, мысалын, қарсы пікірін және қорытындысын қалыптастыруға көмектес.
- Оқушы жазған сөйлемді тексергенде қатесін түсіндір; бүкіл мәтінді оның орнына қайта жазып берме.
- Қазақ тілінде, қысқа, түсінікті, жасөспірімге сай жауап бер.
- Әр жауаптың соңында мүмкін болса 1–3 ойландырушы сұрақ бер.
- Академиялық адалдықты әрдайым сақта.
`;

const READY_TEXT_PATTERNS = [
  /эссе.*жазып бер/i,
  /дайын.*эссе/i,
  /менің орныма.*жаз/i,
  /толық.*мәтін.*жаз/i,
  /көшіріп.*алатындай/i,
  /дайын.*жауап/i
];

function wantsReadyText(text = "") {
  return READY_TEXT_PATTERNS.some((r) => r.test(text));
}

function safeText(value, max = 10000) {
  return String(value ?? "").slice(0, max).trim();
}

async function ensureDataFile() {
  await fs.mkdir(path.dirname(DATA_FILE), { recursive: true });
  try { await fs.access(DATA_FILE); }
  catch { await fs.writeFile(DATA_FILE, "[]", "utf8"); }
}

async function readSubmissions() {
  await ensureDataFile();
  try {
    const raw = await fs.readFile(DATA_FILE, "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

async function writeSubmissions(items) {
  await ensureDataFile();
  await fs.writeFile(DATA_FILE, JSON.stringify(items, null, 2), "utf8");
}

function requireTeacher(req, res, next) {
  const password = req.get("x-teacher-password") || "";
  if (password !== TEACHER_PASSWORD) {
    return res.status(401).json({ error: "Мұғалім құпиясөзі қате." });
  }
  next();
}

app.get("/api/health", (req, res) => {
  res.json({ ok: true, aiConnected: Boolean(ai), model: MODEL });
});

app.post("/api/chat", async (req, res) => {
  try {
    const message = safeText(req.body.message, 4000);
    const essay = safeText(req.body.essay, 12000);
    const topic = safeText(req.body.topic, 500);
    const studentName = safeText(req.body.studentName, 150);
    const history = Array.isArray(req.body.history) ? req.body.history.slice(-10) : [];

    if (!message) return res.status(400).json({ error: "Сұрақ бос болмауы керек." });
    if (!groqApiKey) return res.status(503).json({ error: "Groq API кілті серверге қосылмаған." });

    if (wantsReadyText(message)) {
      return res.json({
        warning: true,
        answer: "Мен сенің орныңа дайын эссе жазбаймын. Бірақ ойыңды өзің құрастыруға көмектесемін. Алдымен: осы тақырып бойынша сенің негізгі пікірің қандай? Оны дәлелдейтін бір нақты мысал келтіре аласың ба?"
      });
    }

    const cleanHistory = history.map((item) => ({
      role: item.role === "model" ? "model" : "user",
      parts: [{ text: safeText(item.text, 2500) }]
    }));

    const groqMessages = [
  { role: "system", content: SYSTEM_INSTRUCTION },
  ...history.map(item => ({
    role: item.role === "model" ? "assistant" : "user",
    content: safeText(item.text, 2500)
  })),
  {
    role: "user",
    content: `Оқушы: ${studentName || "аты көрсетілмеген"}
Тақырып: ${topic || "көрсетілмеген"}
Қазіргі эссе мәтіні:
${essay || "(әлі жазылмаған)"}

Оқушының сұрағы: ${message}`
  }
];

const groqResponse = await fetch("https://api.groq.com/openai/v1/chat/completions", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${groqApiKey}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    model: "openai/gpt-oss-20b",
    messages: groqMessages,
    temperature: 0.45,
    max_completion_tokens: 1000
  })
});

if (!groqResponse.ok) {
  throw new Error(`Groq API қатесі: ${groqResponse.status}`);
}

const groqData = await groqResponse.json();
const answer = groqData.choices?.[0]?.message?.content;

res.json({
  answer: answer || "Жауап алынбады. Сұрағыңды нақтылап көр."
});
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "ЖИ жауабын алу кезінде қате пайда болды." });
  }
});

app.post("/api/evaluate", async (req, res) => {
  try {
    const essay = safeText(req.body.essay, 12000);
    const topic = safeText(req.body.topic, 500);

    if (!essay) {
      return res.status(400).json({ error: "Бағалау үшін эссе мәтінін енгізіңіз." });
    }

    const words = essay.trim().split(/\s+/).filter(Boolean).length;

    if (words < 80) {
      return res.status(400).json({ error: "Эссе кемінде 80 сөз болуы керек." });
    }

    if (!groqApiKey) {
      return res.status(503).json({ error: "Groq API кілті серверге қосылмаған." });
    }

    const prompt = `
Тақырып: ${topic || "көрсетілмеген"}

Оқушы эссесі:
${essay}

Эссені 10 балдық жүйемен бағала.

Әр критерий 0–2 балл:
1) мазмұн және негізгі ой;
2) құрылым және логика;
3) тақырыптың ашылуы;
4) аргумент пен дәлел;
5) тілдік сауаттылық.

Тек JSON форматында жауап бер:

{
  "total": 0,
  "criteria": {
    "content": {"score": 0, "comment": ""},
    "structure": {"score": 0, "comment": ""},
    "topic": {"score": 0, "comment": ""},
    "evidence": {"score": 0, "comment": ""},
    "language": {"score": 0, "comment": ""}
  },
  "strengths": [""],
  "improvements": [""],
  "integrityNote": ""
}

Кері байланыс қазақ тілінде болсын.
Оқушыға дайын сөйлемдер немесе дайын эссе жазып берме.
Тек қатесін түсіндіріп, жақсарту бағытын көрсет.
`;

    const groqResponse = await fetch(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${groqApiKey}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "openai/gpt-oss-20b",
          messages: [
            {
              role: "system",
              content: SYSTEM_INSTRUCTION
            },
            {
              role: "user",
              content: prompt
            }
          ],
          temperature: 0.2,
          max_completion_tokens: 1200
        })
      }
    );

    if (!groqResponse.ok) {
      const details = await groqResponse.text();
      console.error("Groq evaluation error:", groqResponse.status, details);
      return res.status(502).json({
        error: "ЖИ арқылы бағалау мүмкін болмады. Қайта көріңіз."
      });
    }

    const groqData = await groqResponse.json();
    let text = groqData.choices?.[0]?.message?.content || "";

    text = text
      .replace(/```json/gi, "")
      .replace(/```/g, "")
      .trim();

    let result;

    try {
      result = JSON.parse(text);
    } catch (error) {
      console.error("Evaluation JSON parse error:", text);
      return res.status(502).json({
        error: "Бағалау нәтижесін өңдеу мүмкін болмады. Қайта көріңіз."
      });
    }

    result.total = Math.max(
      0,
      Math.min(10, Number(result.total || 0))
    );

    res.json(result);

  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: "Эссені бағалау кезінде қате пайда болды."
    });
  }
});

app.post("/api/submissions", async (req, res) => {
  try {
    const studentName = safeText(req.body.studentName, 150);
    const className = safeText(req.body.className, 80);
    const topic = safeText(req.body.topic, 500);
    const essay = safeText(req.body.essay, 12000);
    const evaluation = req.body.evaluation && typeof req.body.evaluation === "object" ? req.body.evaluation : null;
    const integrityAccepted = Boolean(req.body.integrityAccepted);

    if (!studentName || !className || !topic || !essay) {
      return res.status(400).json({ error: "Аты-жөні, сынып, тақырып және эссе толтырылуы керек." });
    }
    if (!integrityAccepted) {
      return res.status(400).json({ error: "Академиялық адалдық растауын белгілеңіз." });
    }

    const items = await readSubmissions();
    const submission = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      studentName,
      className,
      topic,
      essay,
      evaluation,
      integrityAccepted: true
    };
    items.unshift(submission);
    await writeSubmissions(items.slice(0, 1000));
    res.status(201).json({ ok: true, id: submission.id });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Жұмысты сақтау кезінде қате пайда болды." });
  }
});

app.get("/api/teacher/submissions", requireTeacher, async (req, res) => {
  const items = await readSubmissions();
  res.json(items);
});

app.delete("/api/teacher/submissions/:id", requireTeacher, async (req, res) => {
  const items = await readSubmissions();
  const next = items.filter((x) => x.id !== req.params.id);
  await writeSubmissions(next);
  res.json({ ok: true });
});

app.post("/api/dictionary", async (req, res) => {
  try {
    const text = String(req.body.text || "").trim();
    const words = text.split(/\s+/).filter(Boolean);

    if (!text) {
      return res.status(400).json({ error: "Аударатын сөзді енгізіңіз." });
    }

    if (words.length > 7) {
      return res.status(400).json({
        error: "Сөздік тек 7 сөзге дейінгі сөздер мен сөз тіркестерін аударады."
      });
    }

    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${groqApiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "openai/gpt-oss-20b",
        messages: [
          {
            role: "system",
            content: "Сен орысша-қазақша оқу сөздігісің. Орысша сөзді немесе қысқа сөз тіркесін қазақшаға дәл аудар. Тек қазақша аудармасын бер. Толық сөйлем, абзац немесе эссе жазба."
          },
          { role: "user", content: text }
        ],
        temperature: 0.1
      })
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({ error: "Сөздікті пайдалану кезінде қате шықты." });
    }

    const translation = data.choices?.[0]?.message?.content?.trim();

    res.json({ translation });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Сөздікті пайдалану кезінде қате шықты." });
  }
});
app.use((req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(PORT, () => {
  console.log(`САНА-СЕРІК іске қосылды: http://localhost:${PORT}`);
  if (!apiKey) console.warn("GEMINI_API_KEY орнатылмаған. Чат пен бағалау іске қосылмайды.");
});
