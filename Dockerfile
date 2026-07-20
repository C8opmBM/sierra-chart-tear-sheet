FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY tearsheet ./tearsheet
COPY README.md ./

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

ENV TEARSHEET_INPUT_DIR=/input
ENV TEARSHEET_OUTPUT_DIR=/output

EXPOSE 8080

# Default: run the local web app (upload-and-refresh UI at http://localhost:8080).
#
# For one-off CLI usage instead (no server), override the command, e.g.:
#   docker run --rm -v $(pwd)/input:/input -v $(pwd)/output:/output <image> \
#     tearsheet --input /input/TradeActivityLog_YYYY-MM-DD.txt --output /output/report.html \
#     --starting-balance 50000 --risk-capital 2000
CMD ["python", "-m", "tearsheet.webapp"]
