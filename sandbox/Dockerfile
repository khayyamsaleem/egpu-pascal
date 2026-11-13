# The sandbox the chat's `python` / `shell` tools run in: no network, no host filesystem, capped
# CPU/memory, unprivileged user.  Build once with `bin/local-gpu sandbox --build`.
FROM python:3.12-slim

# The curated set, so common questions ("plot this", "parse that CSV", "solve for x") need no install
# at all.  Anything else goes through deps=[...], which pip-installs into a persistent volume.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy pandas matplotlib scipy sympy tabulate \
        requests beautifulsoup4 lxml pyyaml python-dateutil

RUN useradd -m -u 1000 sandbox && mkdir -p /work /deps && chown -R sandbox:sandbox /work /deps
ENV PYTHONPATH=/deps \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    PIP_CACHE_DIR=/tmp/pip \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /work
USER sandbox
CMD ["python", "-c", "print('egpu-sandbox ready')"]
