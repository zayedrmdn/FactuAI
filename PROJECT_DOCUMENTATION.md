# Project Documentation

## Frontend Architecture

### UI Component Hierarchy

The frontend has been redesigned with a component-based architecture using Next.js App Router.

#### App Shell (`src/app/dashboard/layout.tsx`)
- **Sidebar (`src/components/dashboard/Sidebar.tsx`)**: Collapsible navigation menu.
- **Header (`src/components/dashboard/Header.tsx`)**: Sticky top bar with user profile and page title.
- **Main Content**: Rendered via `children` prop.

#### Feature Modules

1. **Investigation Console (`InputCard.tsx`)**
   - Wraps `InputTabs` for Text, Image, and Video inputs.
   - Uses `FileDropZone` for file uploads.
   - Handles submission logic and validation.

2. **Results Visualization (`ResultsView.tsx`)**
   - **ClaimCard (`ClaimCard.tsx`)**: Displays verdict, confidence score (Progress bar), and evidence (Accordion).
   - **QAResultCard (`QAResultCard.tsx`)**: Displays QA pairs with confidence and sources.
   - **Summary**: Top-level summary card with action buttons (Copy, Share, Export).

3. **Authentication & Profile**
   - **Login (`src/app/login/page.tsx`)**: Secure login form with validation.
   - **Register (`src/app/register/page.tsx`)**: Registration flow.
   - **Profile (`src/app/dashboard/profile/page.tsx`)**: User settings, password management, and avatar upload.

### Design System

The project enforces a strict design system using Tailwind CSS variables defined in `globals.css`.

- **Colors**: Slate (Neutral), Indigo (Primary), Emerald (Success), Red (Destructive).
- **Typography**: Inter font family.
- **Components**: Built on top of `shadcn/ui` patterns (Card, Button, Input, Badge, Accordion, Progress).

### State Management

- **Dashboard Layout State**: Managed locally in `layout.tsx` (sidebar collapse).

## Troubleshooting

- If the backend crashes on Windows with `UnicodeEncodeError` (charmap codec errors) when printing logs, this usually means a message contains non-ASCII characters the console can't render. The backend now sanitizes log output to ASCII with replacement to avoid the crash. To preserve special characters in logs, configure the console to use UTF-8 or redirect logs to a UTF-8 encoded file.
- **Fact-Check State**: Managed via `useFactCheck` hook (preserved).
- **User State**: Persisted in `localStorage` and managed via `useState` in Profile/Auth pages.

### Development Tools

- **React Grab (dev only)**: To assist with AI developer tools and faster context retrieval during development, the React Grab script is injected into the root `layout.tsx` (`frontend/src/app/layout.tsx`) and only loads when `NODE_ENV=development`. It is intentionally omitted in production builds for security and bundle size reasons.
# Project Documentation

## Frontend Architecture

### UI Component Hierarchy

The frontend has been redesigned with a component-based architecture using Next.js App Router.

#### App Shell (`src/app/dashboard/layout.tsx`)
- **Sidebar (`src/components/dashboard/Sidebar.tsx`)**: Collapsible navigation menu.
- **Header (`src/components/dashboard/Header.tsx`)**: Sticky top bar with user profile and page title.
- **Main Content**: Rendered via `children` prop.

#### Feature Modules

1. **Investigation Console (`InputCard.tsx`)**
   - Wraps `InputTabs` for Text, Image, and Video inputs.
   - Uses `FileDropZone` for file uploads.
   - Handles submission logic and validation.

2. **Results Visualization (`ResultsView.tsx`)**
   - **ClaimCard (`ClaimCard.tsx`)**: Displays verdict, confidence score (Progress bar), and evidence (Accordion).
   - **QAResultCard (`QAResultCard.tsx`)**: Displays QA pairs with confidence and sources.
   - **Summary**: Top-level summary card with action buttons (Copy, Share, Export).

3. **Authentication & Profile**
   - **Login (`src/app/login/page.tsx`)**: Secure login form with validation.
   - **Register (`src/app/register/page.tsx`)**: Registration flow.
   - **Profile (`src/app/dashboard/profile/page.tsx`)**: User settings, password management, and avatar upload.

### Design System

The project enforces a strict design system using Tailwind CSS variables defined in `globals.css`.


```

### Development Tools

- **React Grab (dev only)**: To assist with AI developer tools and faster context retrieval during development, the React Grab script is injected into the root `layout.tsx` (`frontend/src/app/layout.tsx`) and only loads when `NODE_ENV=development`. It is intentionally omitted in production builds for security and bundle size reasons.


- **Dashboard Layout State**: Managed locally in `layout.tsx` (sidebar collapse).
