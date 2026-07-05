web: prisma generate --schema=prisma/schema.prisma && prisma db push --schema=prisma/schema.prisma && cd backend && gunicorn app:app --bind 0.0.0.0:$PORT
