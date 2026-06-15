# TumorDetect
Web application using AI to detect brain tumors from MRI scans.

## Setup

1. Copy `.env.example` to `.env` and fill in `SECRET_KEY`, `DATABASE_URL`, mail credentials, and optional admin seed values.
2. Create a PostgreSQL database and point `DATABASE_URL` at it.
3. Install dependencies from `requirements.txt`.
4. Generate the Prisma client and push the schema to Postgres with `prisma generate` and `prisma db push`.
5. Start the Flask app from the `backend` directory.

The app now uses Prisma ORM for `User` and `MriUpload` records, so sign up, sign in, profile updates, password resets, and admin access all read and write the Postgres database.
