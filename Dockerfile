# The scorer builds this on linux/amd64 and runs it with no network.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# sqlglot source at the pinned commit (read at run time to check what a patch changes).
ARG SQLGLOT_SHA=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
RUN python -c "import urllib.request,tarfile,io,sys; \
data=urllib.request.urlopen('https://github.com/tobymao/sqlglot/archive/${SQLGLOT_SHA}.tar.gz', timeout=120).read(); \
t=tarfile.open(fileobj=io.BytesIO(data)); \
root=t.getmembers()[0].name.split('/')[0]; \
ms=[m for m in t.getmembers() if m.name.split('/')[1:2] in (['sqlglot'],['tests'],['pyproject.toml'],['setup.cfg'],['setup.py'])]; \
[setattr(m,'name',m.name.split('/',1)[1]) for m in ms]; \
t.extractall('/app/src', members=ms, filter='data')" \
 && test -f /app/src/sqlglot/parser.py && chmod -R a+rX /app

# Bytecode for the pinned source, so the import check (selector.py) does not
# recompile sqlglot on every run in the read-only container.
RUN python -m compileall -q /app/src/sqlglot && chmod -R a+rX /app/src

# pytz (PyPI, pinned): tests/dialects/test_bigquery.py imports it, so without it a
# check run that includes BigQuery tests cannot import them and proves nothing.
RUN pip install --no-cache-dir pytz==2024.2

# Structure maps (def/class/statement spans, import graph) from the pinned source.
COPY build_struct.py /app/build_struct.py
RUN python /app/build_struct.py /app/src /app/struct

# Coverage and fixture maps built once at the pinned commit (see README).
COPY maps /app/maps
COPY selector.py /app/selector.py
RUN chmod -R a+rX /app
CMD ["python", "-S", "-E", "/app/selector.py"]
