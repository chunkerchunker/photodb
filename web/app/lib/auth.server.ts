import crypto from "node:crypto";
import { redirect } from "react-router";
import {
  createSession,
  deleteSession,
  getUserBySessionToken,
  getUserByUsername,
  updateUserPasswordHash,
  type AppUser,
} from "./db.server";

const SESSION_COOKIE_NAME = "photodb_session";
const SESSION_TTL_DAYS = 30;

function parseCookies(cookieHeader: string | null): Record<string, string> {
  if (!cookieHeader) return {};
  return cookieHeader.split(";").reduce((acc, part) => {
    const [rawKey, ...rawValue] = part.trim().split("=");
    if (!rawKey) return acc;
    acc[decodeURIComponent(rawKey)] = decodeURIComponent(rawValue.join("="));
    return acc;
  }, {} as Record<string, string>);
}

function buildSetCookie(token: string, expiresAt: Date): string {
  const expires = expiresAt.toUTCString();
  return `${SESSION_COOKIE_NAME}=${encodeURIComponent(token)}; Path=/; HttpOnly; SameSite=Lax; Expires=${expires}`;
}

function buildClearCookie(): string {
  return `${SESSION_COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;
}

async function scryptHash(password: string, salt: Buffer): Promise<Buffer> {
  return await new Promise((resolve, reject) => {
    crypto.scrypt(password, salt, 64, (err, derivedKey) => {
      if (err) reject(err);
      else resolve(derivedKey as Buffer);
    });
  });
}

export async function hashPassword(password: string): Promise<string> {
  const salt = crypto.randomBytes(16);
  const derivedKey = await scryptHash(password, salt);
  return `scrypt$${salt.toString("hex")}$${derivedKey.toString("hex")}`;
}

export async function verifyPassword(password: string, storedHash: string): Promise<boolean> {
  if (!storedHash.startsWith("scrypt$")) {
    return password === storedHash;
  }
  const parts = storedHash.split("$");
  if (parts.length !== 3) return false;
  const salt = Buffer.from(parts[1], "hex");
  const expected = Buffer.from(parts[2], "hex");
  const derivedKey = await scryptHash(password, salt);
  return crypto.timingSafeEqual(expected, derivedKey);
}

export async function getSessionUser(request: Request): Promise<AppUser | null> {
  const cookies = parseCookies(request.headers.get("Cookie"));
  const token = cookies[SESSION_COOKIE_NAME];
  if (!token) return null;
  return await getUserBySessionToken(token);
}

export async function requireUser(request: Request): Promise<AppUser> {
  const user = await getSessionUser(request);
  if (!user) {
    throw redirect("/login");
  }
  return user;
}

export async function createLoginSession(userId: number): Promise<{ token: string; cookie: string }> {
  const token = crypto.randomBytes(32).toString("hex");
  const expiresAt = new Date(Date.now() + SESSION_TTL_DAYS * 24 * 60 * 60 * 1000);
  await createSession(userId, token, expiresAt);
  return { token, cookie: buildSetCookie(token, expiresAt) };
}

export async function handleLogin(request: Request) {
  const form = await request.formData();
  const username = String(form.get("username") || "").trim();
  const password = String(form.get("password") || "");

  if (!username || !password) {
    return { error: "Username and password are required." };
  }

  const user = await getUserByUsername(username);
  if (!user) {
    return { error: "Invalid username or password." };
  }

  const ok = await verifyPassword(password, user.password_hash);
  if (!ok) {
    return { error: "Invalid username or password." };
  }

  if (!user.password_hash.startsWith("scrypt$")) {
    const upgraded = await hashPassword(password);
    await updateUserPasswordHash(user.id, upgraded);
  }

  const { cookie } = await createLoginSession(user.id);
  throw redirect("/", {
    headers: {
      "Set-Cookie": cookie,
    },
  });
}

export async function handleLogout(request: Request) {
  const cookies = parseCookies(request.headers.get("Cookie"));
  const token = cookies[SESSION_COOKIE_NAME];
  if (token) {
    await deleteSession(token);
  }
  throw redirect("/login", {
    headers: {
      "Set-Cookie": buildClearCookie(),
    },
  });
}

export function getClearSessionCookie(): string {
  return buildClearCookie();
}
