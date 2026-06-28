ARG NGINX_IMAGE=nginx:stable-alpine

FROM ${NGINX_IMAGE}

LABEL org.opencontainers.image.title="DataAgent Analyst Frontend"
LABEL org.opencontainers.image.description="Nginx frontend for DataAgent Analyst"

RUN rm -f /etc/nginx/conf.d/default.conf

COPY deploy/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY frontend /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK \
  --interval=15s \
  --timeout=5s \
  --start-period=10s \
  --retries=5 \
  CMD wget -q -O - http://127.0.0.1/healthz >/dev/null || exit 1
