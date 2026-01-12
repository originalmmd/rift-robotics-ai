// src/index.ts
// Rift Voice Gateway (Cloudflare Worker)
// Flow: Device -> STT (OpenAI) -> intent match (responses.json) -> TTS (OpenAI WAV) -> EQ -> return audio+actions

export interface Env {
	// Secrets (wrangler secret put ...)
	OPENAI_API_KEY: string;
	BOT_SHARED_SECRET: string;

	// Vars (wrangler.jsonc "vars")
	RESPONSES_URL: string;
	DEFAULT_FX: string;

	// Optional vars
	STT_MODEL?: string; // default: "whisper-1"
	TTS_MODEL?: string; // default: "gpt-4o-mini-tts"
	TTS_VOICE?: string; // default: "cedar"
}

type IntentDef = {
	id: string;
	match: { any?: string[]; regex?: string };
	reply: string[];
	voice_fx?: string;
	actions?: any;
};

type ResponsesFile = {
	version: number;
	intents: IntentDef[];
	fallback: { reply: string[]; voice_fx?: string; actions?: any };
};

const CACHE_TTL_MS = 60_000; // 1 minute
let cachedResponses: { at: number; data: ResponsesFile } | null = null;

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);

		// CORS (handy for Android dev)
		if (request.method === 'OPTIONS') {
			return new Response(null, {
				status: 204,
				headers: corsHeaders(),
			});
		}

		try {
			if (request.method === 'GET' && url.pathname === '/health') {
				return json({ ok: true }, 200, corsHeaders());
			}

			if (request.method === 'POST' && url.pathname === '/v1/command') {
				// --- Auth (MVP): shared secret ---
				const botToken = request.headers.get('X-Bot-Token') || '';
				if (!botToken || botToken !== env.BOT_SHARED_SECRET) {
					return json({ error: 'unauthorized' }, 401, corsHeaders());
				}

				const includeAudio = url.searchParams.get('audio') === '1';

				// --- Parse multipart form ---
				const ct = request.headers.get('content-type') || '';
				if (!ct.includes('multipart/form-data')) {
					return json({ error: 'expected multipart/form-data' }, 400, corsHeaders());
				}

				const form = await request.formData();
				const audio = form.get('audio');
				if (!(audio instanceof File)) {
					return json({ error: "missing audio file field 'audio'" }, 400, corsHeaders());
				}

				// --- STT (OpenAI) ---
				const transcript = await openaiStt(audio, env);

				// --- Responses + Intent match ---
				const responses = await loadResponses(env);
				const intent = matchIntent(transcript, responses);

				const replyText = pickRandom(intent.reply);
				const fx = intent.voice_fx || env.DEFAULT_FX || 'archive';
				const actions = intent.actions ?? null;

				// --- TTS (OpenAI) WAV ---
				const wav = await openaiTtsWav(replyText, env);

				// --- Apply lore EQ (archive band-pass) ---
				const processedWav = applyFxPreset(wav, fx);

				// --- Return ---
				return json(
					{
						transcript,
						intent: intent.id,
						reply_text: replyText,
						reply_audio_url: `${url.origin}/v1/command.wav`,
						reply_audio: includeAudio ? { mime: 'audio/wav', base64: arrayBufferToBase64(processedWav) } : null,
						actions,
					},
					200,
					corsHeaders()
				);
			}

			if (request.method === 'POST' && url.pathname === '/v1/command.wav') {
				// --- Auth (MVP): shared secret ---
				const botToken = request.headers.get('X-Bot-Token') || '';
				if (!botToken || botToken !== env.BOT_SHARED_SECRET) {
					return new Response('unauthorized', { status: 401, headers: corsHeaders() });
				}

				// --- Parse multipart form ---
				const ct = request.headers.get('content-type') || '';
				if (!ct.includes('multipart/form-data')) {
					return new Response('expected multipart/form-data', { status: 400, headers: corsHeaders() });
				}

				const form = await request.formData();
				const audio = form.get('audio');
				if (!(audio instanceof File)) {
					return new Response("missing audio file field 'audio'", { status: 400, headers: corsHeaders() });
				}

				// --- STT ---
				const transcript = await sttOpenAI(audio, env);

				// --- Load intents + match ---
				const responses = await loadResponses(env);
				const intent = matchIntent(transcript, responses);

				const replyText = pickRandom(intent.reply);
				const preset = (intent.voice_fx || env.DEFAULT_FX || 'archive').toLowerCase();

				// --- TTS -> WAV ---
				const wav = await openaiTtsWav(replyText, env);

				// --- Optional FX (returns WAV ArrayBuffer) ---
				const processed = applyFxPreset(wav, preset);

				return new Response(processed, {
					status: 200,
					headers: {
						...corsHeaders(),
						'content-type': 'audio/wav',
						'cache-control': 'no-store',
					},
				});
			}

			return json({ error: 'not found' }, 404, corsHeaders());
		} catch (err: any) {
			// Return JSON instead of Cloudflare 1101
			return json(
				{
					error: 'worker_exception',
					message: String(err?.message || err),
				},
				500,
				corsHeaders()
			);
		}
	},
};

// -------------------- OpenAI STT --------------------
async function openaiStt(file: File, env: Env): Promise<string> {
	const fd = new FormData();
	fd.append('file', file, file.name);

	// Safe baseline; change via env.STT_MODEL if you want later
	fd.append('model', env.STT_MODEL || 'whisper-1');

	const res = await fetch('https://api.openai.com/v1/audio/transcriptions', {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${env.OPENAI_API_KEY}`,
		},
		body: fd,
	});

	if (!res.ok) {
		const txt = await res.text();
		throw new Error(`OpenAI STT failed: ${res.status} ${txt}`);
	}

	const data = (await res.json()) as any;
	return (data.text || '').toString().trim();
}

// -------------------- OpenAI TTS (WAV) --------------------
async function openaiTtsWav(text: string, env: Env): Promise<ArrayBuffer> {
	const payload = {
		model: env.TTS_MODEL || 'gpt-4o-mini-tts',
		voice: env.TTS_VOICE || 'onyx',
		input: text,
		response_format: 'wav',
		instructions:
			'Voice: older, calm, low energy, slightly distant. ' +
			'Tone: restrained, archival reconstruction. ' +
			'Pace: slower than normal. ' +
			'No cheeriness, no salesy assistant tone. Minimal emotion.',
	};

	const res = await fetch('https://api.openai.com/v1/audio/speech', {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${env.OPENAI_API_KEY}`,
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(payload),
	});

	if (!res.ok) throw new Error(`OpenAI TTS failed: ${res.status} ${await res.text()}`);
	return await res.arrayBuffer();
}

// -------------------- Responses JSON fetch + cache --------------------
async function loadResponses(env: Env): Promise<ResponsesFile> {
	const now = Date.now();
	if (cachedResponses && now - cachedResponses.at < CACHE_TTL_MS) return cachedResponses.data;

	if (!env.RESPONSES_URL) {
		throw new Error('RESPONSES_URL is not set');
	}

	const res = await fetch(env.RESPONSES_URL, {
		headers: { 'cache-control': 'no-cache' },
	});

	if (!res.ok) {
		throw new Error(`Failed to fetch responses.json: ${res.status} ${await res.text()}`);
	}

	const data = (await res.json()) as ResponsesFile;
	console.log('RESPONSES_URL:', env.RESPONSES_URL);
	console.log('RESPONSES_VERSION:', data.version);
	console.log(
		'INTENTS:',
		data.intents?.map((i) => i.id)
	);

	cachedResponses = { at: now, data };
	return data;
}

// -------------------- Intent matching --------------------
function normalize(s: string) {
	return s
		.toLowerCase()
		.replace(/[^\p{L}\p{N}\s]+/gu, '') // strip punctuation safely
		.replace(/\s+/g, ' ')
		.trim();
}

function matchIntent(transcript: string, rf: ResponsesFile): IntentDef {
	const t = normalize(transcript);

	for (const intent of rf.intents) {
		const any = intent.match.any?.map(normalize) || [];
		if (any.some((p) => p && t.includes(p))) return intent;

		if (intent.match.regex) {
			try {
				const re = new RegExp(intent.match.regex, 'i');
				if (re.test(transcript)) return intent;
			} catch {
				// ignore bad regex
			}
		}
	}

	// fallback mapped into an IntentDef-ish object
	return {
		id: 'FALLBACK',
		match: {},
		reply: rf.fallback.reply,
		voice_fx: rf.fallback.voice_fx,
		actions: rf.fallback.actions,
	};
}

function pickRandom<T>(arr: T[]): T {
	return arr[Math.floor(Math.random() * arr.length)];
}

// -------------------- FX: Audacity-like archive band-pass --------------------
function applyFxPreset(wav: ArrayBuffer, preset: string): ArrayBuffer {
	// For now: only "archive" preset. Anything else returns raw WAV.
	if (preset !== 'archive') return wav;

	const decoded = decodeWavPcm16Mono(wav);
	if (!decoded) return wav;

	const { sampleRate, samples } = decoded;

	// Approx your Audacity curve:
	// - aggressive low cut
	// - flat midband
	// - steep-ish high cut
	highPass1(samples, sampleRate, 500);
	lowPass1(samples, sampleRate, 3000);
	gainDb(samples, -1.0);

	return encodeWavPcm16Mono(sampleRate, samples);
}

// -------------------- WAV decode/encode (PCM16 mono only) --------------------
function decodeWavPcm16Mono(buf: ArrayBuffer): { sampleRate: number; samples: Float32Array } | null {
	const dv = new DataView(buf);
	const u8 = new Uint8Array(buf);

	const readStr = (off: number, len: number) => String.fromCharCode(...u8.slice(off, off + len));

	if (dv.byteLength < 44) return null;
	if (readStr(0, 4) !== 'RIFF' || readStr(8, 4) !== 'WAVE') return null;

	let off = 12;
	let fmtOff = -1;
	let dataOff = -1;
	let dataSize = 0;

	while (off + 8 <= dv.byteLength) {
		const id = readStr(off, 4);
		const size = dv.getUint32(off + 4, true);
		const chunkStart = off + 8;

		if (id === 'fmt ') fmtOff = chunkStart;
		if (id === 'data') {
			dataOff = chunkStart;
			dataSize = size;
			break;
		}

		off = chunkStart + size + (size % 2);
	}

	if (fmtOff < 0 || dataOff < 0) return null;

	const audioFormat = dv.getUint16(fmtOff + 0, true);
	const numChannels = dv.getUint16(fmtOff + 2, true);
	const sampleRate = dv.getUint32(fmtOff + 4, true);
	const bitsPerSample = dv.getUint16(fmtOff + 14, true);

	if (audioFormat !== 1 || numChannels !== 1 || bitsPerSample !== 16) return null;

	const sampleCount = dataSize / 2;
	if (dataOff + dataSize > dv.byteLength) return null;

	const samples = new Float32Array(sampleCount);
	let p = dataOff;

	for (let i = 0; i < sampleCount; i++) {
		const s = dv.getInt16(p, true);
		samples[i] = s / 32768;
		p += 2;
	}

	return { sampleRate, samples };
}

function encodeWavPcm16Mono(sampleRate: number, samples: Float32Array): ArrayBuffer {
	const dataSize = samples.length * 2;
	const buf = new ArrayBuffer(44 + dataSize);
	const dv = new DataView(buf);
	const u8 = new Uint8Array(buf);

	const writeStr = (off: number, s: string) => {
		for (let i = 0; i < s.length; i++) u8[off + i] = s.charCodeAt(i);
	};

	writeStr(0, 'RIFF');
	dv.setUint32(4, 36 + dataSize, true);
	writeStr(8, 'WAVE');
	writeStr(12, 'fmt ');
	dv.setUint32(16, 16, true); // fmt chunk size
	dv.setUint16(20, 1, true); // PCM
	dv.setUint16(22, 1, true); // mono
	dv.setUint32(24, sampleRate, true);
	dv.setUint32(28, sampleRate * 2, true); // byte rate
	dv.setUint16(32, 2, true); // block align
	dv.setUint16(34, 16, true); // bits
	writeStr(36, 'data');
	dv.setUint32(40, dataSize, true);

	let p = 44;
	for (let i = 0; i < samples.length; i++) {
		let x = Math.max(-1, Math.min(1, samples[i]));
		const s = (x * 32767) | 0;
		dv.setInt16(p, s, true);
		p += 2;
	}

	return buf;
}

// -------------------- Simple DSP blocks --------------------
function gainDb(x: Float32Array, db: number) {
	const g = Math.pow(10, db / 20);
	for (let i = 0; i < x.length; i++) x[i] *= g;
}

// 1st-order RC high-pass
function highPass1(x: Float32Array, fs: number, fc: number) {
	const dt = 1 / fs;
	const rc = 1 / (2 * Math.PI * fc);
	const a = rc / (rc + dt);

	let yPrev = 0;
	let xPrev = x[0] || 0;
	for (let i = 0; i < x.length; i++) {
		const y = a * (yPrev + x[i] - xPrev);
		xPrev = x[i];
		yPrev = y;
		x[i] = y;
	}
}

// 1st-order RC low-pass
function lowPass1(x: Float32Array, fs: number, fc: number) {
	const dt = 1 / fs;
	const rc = 1 / (2 * Math.PI * fc);
	const a = dt / (rc + dt);

	let y = x[0] || 0;
	for (let i = 0; i < x.length; i++) {
		y = y + a * (x[i] - y);
		x[i] = y;
	}
}

// -------------------- Base64 + JSON + CORS helpers --------------------
function arrayBufferToBase64(buf: ArrayBuffer): string {
	const bytes = new Uint8Array(buf);
	let binary = '';
	const chunk = 0x8000;
	for (let i = 0; i < bytes.length; i += chunk) {
		binary += String.fromCharCode(...bytes.slice(i, i + chunk));
	}
	return btoa(binary);
}

function json(data: any, status = 200, extraHeaders?: HeadersInit): Response {
	const headers: HeadersInit = {
		'content-type': 'application/json; charset=utf-8',
		...(extraHeaders || {}),
	};
	return new Response(JSON.stringify(data), { status, headers });
}

function corsHeaders(): HeadersInit {
	return {
		'Access-Control-Allow-Origin': '*',
		'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
		'Access-Control-Allow-Headers': 'Content-Type,X-Bot-Id,X-Bot-Token',
	};
}
async function sttOpenAI(audio: File, env: Env): Promise<string> {
	const fd = new FormData();
	fd.append('file', audio, audio.name);
	fd.append('model', env.STT_MODEL || 'whisper-1');

	const res = await fetch('https://api.openai.com/v1/audio/transcriptions', {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${env.OPENAI_API_KEY}`,
		},
		body: fd,
	});

	if (!res.ok) {
		const txt = await res.text();
		throw new Error(`OpenAI STT failed: ${res.status} ${txt}`);
	}

	const data = (await res.json()) as any;
	return (data.text || '').toString().trim();
}
