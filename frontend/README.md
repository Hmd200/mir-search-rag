# MIR Search frontend

Admin dashboard and search UI for the dual-engine retrieval API.

## Install

From this directory:

```bash
npm install
```

## Run

Start the FastAPI backend first. It listens on `http://127.0.0.1:8000` by default.

Then start the Vite dev server:

```bash
npm run dev
```

The UI is at `http://localhost:5173`. Vite proxies `/api` to the backend, so the browser does not need to call the API origin directly.

- Search: `http://localhost:5173/`
- Admin dashboard: `http://localhost:5173/admin`

## Build

```bash
npm run build
npm run preview
```

`preview` serves the production bundle (default port 4173). The backend already allows that origin in CORS.
