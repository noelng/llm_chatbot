FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

RUN chmod +x /app/entrypoint.sh

CMD ["sh", "entrypoint.sh"]