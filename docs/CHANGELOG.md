# Changelog

All notable changes to this project will be documented in this file.

## [4.0.2] - 2025-12-14

### New Default Model: Meta Llama 3.3 70B Instruct

Upgraded the default verification model to Meta's latest Llama 3.3 70B Instruct, offering superior reasoning at a low cost.

#### Added

- **New Model: Llama 3.3 70B** - Added `meta-llama/llama-3.3-70b-instruct` to the OpenRouter model registry:
  - 131K context window
  - $0.10/M input tokens, $0.32/M output tokens
  - Superior reasoning capabilities for fact verification
  - Marked as recommended and default

#### Changed

- **Default Provider** - Changed from NVIDIA to OpenRouter for better model availability
- **Default Model** - Changed from `nvidia-qwen2.5-7b` to `openrouter-llama-3.3-70b`
- **Pipeline Reasoning Model** - Updated from `openrouter-tongyi-deepresearch-30b` to `openrouter-llama-3.3-70b`
- **Backend Settings** - Updated `OPENROUTER_MODEL` default from `anthropic/claude-3-haiku` to `meta-llama/llama-3.3-70b-instruct`

#### Files Modified

- `frontend/src/config/ai-models.ts` - Added new model, updated defaults
- `frontend/src/stores/pipeline-models-store.ts` - Updated reasoning default
- `backend/app/core/settings.py` - Updated OpenRouter model default

---

## [4.0.1] - 2025-12-14

### Frontend V4 Backend Synchronization

This release synchronizes the frontend with the new V4 backend architecture.

#### Changed

- **API Configuration** - Updated all API endpoints from legacy Flask (port 5000) to V4 FastAPI (port 8000):
  - `frontend/src/lib/api.ts`
  - `frontend/src/lib/api-dashboard.ts`
  - `frontend/src/lib/dashboard/constants.ts`
  - `frontend/src/lib/hooks/useFactCheck.ts` (changed `/api/process` → `/api/analyze`)
  - `frontend/src/app/register/page.tsx`
  - `frontend/src/components/UserAvatar.tsx`
  - `frontend/src/components/dashboard/Header.tsx`

- **Next.js Config** - Added `127.0.0.1` to allowed image hosts for profile pictures

#### Fixed

- **Logout Functionality** - All logout handlers now properly:
  - Remove both `token` and `user` from localStorage
  - Clear the `useUser` hook cache to prevent stale state
  - Files updated: `Sidebar.tsx`, `LandingNav.tsx`

- **Login Error Display** - Added inline error alert (red box with icon) in addition to toast notifications for failed login attempts

- **Ghost Endpoint Errors** - Removed calls to non-existent V4 endpoints:
  - `/api/profile/:id` - Now uses localStorage directly (V4 doesn't have profile endpoint)
  - `/api/validate` - Now uses client-side validation only (V4's `/api/analyze` handles its own validation)

#### Removed

- Removed blocking API calls in `useUser.ts` that caused dashboard crashes
- Removed LLM validation call in `useInputValidation.ts` that caused 404 errors

### Technical Notes

- Next.js API rewrites in `next.config.ts` proxy all `/api/*` requests to `http://127.0.0.1:8000/api/*`
- The V4 backend exposes only: `GET /health`, `POST /api/analyze`, `POST /api/login`
- Profile management features are disabled until backend implements `/api/profile` endpoints

### Migration Steps

1. Clear Next.js cache: `rm -rf frontend/.next`
2. Restart the dev server: `npm run dev`
3. Ensure backend is running on port 8000: `uvicorn app.main:app --reload`
