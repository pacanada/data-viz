FROM python:3.12-slim-bookworm as build
ENV APP_DIR=/app
WORKDIR $APP_DIR
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install dependencies
COPY .python-version .
COPY uv.lock .
COPY pyproject.toml .
RUN uv sync --frozen

FROM build AS final
# Copy the code
WORKDIR $APP_DIR
# Copy the rest of the application code
COPY 00_Intro.py 00_Intro.py
COPY pages pages
COPY .streamlit .streamlit
COPY .streamlit/config.toml .streamlit/config.toml



# Copy the uv binary again for runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
# Copy the entire application (dependencies + code) from the build stage
COPY --from=build $APP_DIR $APP_DIR



EXPOSE 8501
CMD ["uv", "run", "python", "-m", "streamlit", "run", "00_Intro.py"]