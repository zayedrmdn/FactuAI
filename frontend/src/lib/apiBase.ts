const FALLBACK_ORIGIN = 'http://127.0.0.1:8000';

function normalizeBackendOrigin(raw: string): string {
  let origin = (raw || '').trim();
  if (!origin) origin = FALLBACK_ORIGIN;

  // Remove trailing slash
  origin = origin.replace(/\/$/, '');

  // Support older env var formats pointing at API routes
  origin = origin.replace(/\/api\/analyze$/, '');
  origin = origin.replace(/\/api$/, '');

  return origin;
}

export function getBackendOrigin(): string {
  return normalizeBackendOrigin(process.env.NEXT_PUBLIC_API_URL || FALLBACK_ORIGIN);
}

export function getApiBaseUrl(): string {
  return `${getBackendOrigin()}/api`;
}

export function getApiUrl(pathname: string): string {
  const path = pathname.startsWith('/') ? pathname.slice(1) : pathname;
  return `${getApiBaseUrl()}/${path}`;
}

export function getMediaUrl(pathname: string): string {
  if (!pathname) return getBackendOrigin();
  return `${getBackendOrigin()}${pathname.startsWith('/') ? pathname : `/${pathname}`}`;
}
