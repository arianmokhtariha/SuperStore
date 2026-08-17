"""
PostgreSQL project bootstrapper.

Creates a PostgreSQL database, runs a directory of versioned .sql files
against it, and optionally loads CSV files into the resulting tables.

Reusable across projects: to retarget it, edit ONLY the CONFIG block below.
Nothing further down is project-specific.

It covers two project shapes, and the difference between them is entirely
a matter of what CONFIG says:

  1. SQL schema + CSV data.  The .sql files define empty tables; the rows
     arrive from CSV files through COPY.
         SQL_DIR = _ROOT / "sql"
         CSV_MAPPING = {"raw_data/clubs.csv": "clubs", ...}

  2. Pure SQL.  One folder holds everything — DDL, seed/INSERT data, views,
     functions, whatever else — and the whole database is built by running
     those files in order. There are no CSVs at all.
         SQL_DIR = _ROOT / "sql" / "oltp"
         CSV_MAPPING = {}
     In this shape pandas is never imported and LOAD_MODE / CSV_DIALECT are
     inert, so the only dependency is psycopg2.

Either way the .sql files run **verbatim**, one batch per file, in filename
order — so number them to control that order (00_schema.sql, 10_data.sql,
20_views.sql). Ordering is the one thing the script cannot infer for you;
it is printed for review before anything runs.

Usage:
    python db_setup.py
"""

from __future__ import annotations

import csv
import io
import itertools
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg2
from psycopg2 import extensions, sql

from db_config import DB_CONFIG, DB_CONFIG_INIT

if TYPE_CHECKING:   # pandas is imported lazily, only by the "pandas" CSV mode
    import pandas as pd

_ROOT = Path(__file__).parent

# ============================================================
# CONFIG — the only block to edit per project
# ============================================================

# Every *.sql file directly inside this directory is executed in filename
# order, each as a single batch, exactly as written. Number them to control
# the order: 00_schema.sql, 10_data.sql, 20_views.sql
#
# The files may contain anything Postgres accepts — DDL, INSERTs, COPY from
# a server-side file, views, functions. A folder that builds an entire data
# model on its own is the intended use, not a special case.
#
# Sub-directories are deliberately NOT searched, so sibling folders can hold
# SQL that belongs to a later stage (marts, tests) without being swept into
# the bootstrap. Point this at the exact folder to run:
#     _ROOT / "sql"           → runs sql/*.sql
#     _ROOT / "sql" / "oltp"  → runs sql/oltp/*.sql, ignores sql/marts/
#
# None skips the SQL step entirely (for a project whose tables already exist).
# sql/OLTP holds the raw source model, split out of the original Superstore.sql
# export. sql/OLAP (the star schema built on top of it) is a separate stage and
# is deliberately not swept in here — point SQL_DIR at it, with RECREATE_DB
# False, to build that layer on top of a loaded OLTP.
SQL_DIR: Path | None = _ROOT / "sql" / "OLTP"

# {csv path relative to this file: target table}
# (Important) Order parent tables before child tables so foreign keys resolve.
# Table names may be schema-qualified, e.g. "raw.sales".
# Leave it EMPTY ({}) when the data ships inside the .sql files instead.
CSV_MAPPING: dict[str, str] = {}

# Applies to CSV_MAPPING only; ignored when there are no CSVs.
# "stream" — pipe the CSV straight into Postgres and let Postgres parse it.
#            Values land byte-exact and memory stays flat on huge files.
# "pandas" — read into a DataFrame first. Only needed when the data has to
#            be inspected or transformed in Python before it lands.
LOAD_MODE: str = "stream"

# True  — drop the database and rebuild it from scratch.
# False — leave the database in place and run everything on top of it. The
#         .sql files must then be re-runnable (CREATE TABLE IF NOT EXISTS,
#         CREATE OR REPLACE VIEW, DROP ... IF EXISTS) or they will fail on
#         objects that already exist.
RECREATE_DB: bool = True

# Applies to CSV_MAPPING only; ignored when there are no CSVs.
# "null_token" is the text that means NULL in the source files. An empty
# string (the default) means an empty CSV field is NULL.
CSV_DIALECT: dict[str, str] = {
    "delimiter":  ",",
    "encoding":   "utf-8",
    "null_token": "",
}

# ============================================================
# ANSI COLOR HELPERS (no external deps)
# ============================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"


def _c(text: object, *codes: str) -> str:
    """Wrap text with one or more ANSI codes and reset at the end."""
    return "".join(codes) + str(text) + RESET


# ============================================================
# SPINNER (pure threading, no external deps)
# ============================================================

class Spinner:
    """
    Context-manager spinner that animates on a single terminal line while a
    blocking operation runs, then replaces itself with a success / failure
    icon when complete.

    Usage:
        with Spinner("Creating table players"):
            do_slow_work()
        # success → ✅  Creating table players
        # error   → ❌  Creating table players  (failed)
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    INTERVAL = 0.08   # seconds between frames

    def __init__(self, message: str, indent: int = 2) -> None:
        self.message = message
        self.indent = " " * indent
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(
                f"\r{self.indent}{_c(frame, CYAN)}  {self.message}{_c('...', DIM)}"
            )
            sys.stdout.flush()
            time.sleep(self.INTERVAL)

    def start(self) -> "Spinner":
        self._thread.start()
        return self

    def stop(self, success: bool = True, suffix: str = "") -> None:
        self._stop.set()
        self._thread.join()
        icon = _c("✅", GREEN) if success else _c("❌", RED)
        tag = _c(f"  ({suffix})", DIM) if suffix else ""
        sys.stdout.write(f"\r{self.indent}{icon}  {self.message}{tag}\n")
        sys.stdout.flush()

    # ── context-manager protocol ─────────────────────────────────────────
    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        # exc_type is None when the block exits cleanly
        self.stop(success=(exc_type is None))
        # Do NOT suppress exceptions — let them propagate to the caller
        return False


# ============================================================
# TERMINAL OUTPUT
# ============================================================

_BOX_WIDTH = 52


def _print_header(title: str) -> None:
    """Print a bold section header with a surrounding box."""
    bar = "═" * _BOX_WIDTH
    print()
    print(_c(f"╔{bar}╗", BOLD, CYAN))
    print(_c(f"║  {title:<{_BOX_WIDTH - 2}}║", BOLD, CYAN))
    print(_c(f"╚{bar}╝", BOLD, CYAN))
    print()


def _print_box(title: str, body: list[str], color: str) -> None:
    """
    Print a titled, boxed message block in the given ANSI colour.

    Padding is computed with len(), so every character placed inside the box
    must span the same number of terminal cells as code points. ⚠ ℹ ✔ ✖ are
    safe; ✅ ❌ are not (one code point, two cells) — they belong on the
    unpadded spinner lines instead.
    """
    inner = _BOX_WIDTH - 2
    bar = "═" * _BOX_WIDTH
    print(_c(f"  ╔{bar}╗", BOLD, color))
    print(_c(f"  ║  {title:<{inner}}║", BOLD, color))
    if body:
        print(_c(f"  ╠{bar}╣", BOLD, color))
        for line in body:
            print(_c(f"  ║  {line:<{inner}}║", color))
    print(_c(f"  ╚{bar}╝", BOLD, color))


def _print_section(label: str) -> None:
    """Print a dimmed step sub-header."""
    print(_c(f"\n  ── {label} {'─' * max(0, 46 - len(label))}", DIM))


def _human_size(num_bytes: int) -> str:
    """Render a byte count as a short, human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} TB"


def _plural(count: int, singular: str, plural: str = "") -> str:
    """Render a count with the right form of its noun."""
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count:,} {word}"


# ============================================================
# DATABASE LIFECYCLE
# ============================================================

def _database_exists(db_name: str, config_init: dict[str, str]) -> bool:
    """Return True if the target database already exists in the cluster."""
    conn = psycopg2.connect(**config_init)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", [db_name])
        return cursor.fetchone() is not None
    finally:
        cursor.close()
        conn.close()


_PLAN_MAX_FILES = 14


def _plan_lines(
    files: list[Path],
    csv_mapping: dict[str, str],
    missing_csvs: list[str],
) -> list[str]:
    """
    Describe what is about to be executed, in execution order.

    Filename order is the only thing deciding whether foreign keys resolve
    and whether data lands after the tables that hold it, and it is the one
    thing this script cannot infer. Showing the resolved order up front turns
    a mis-numbered file into something spotted before the run rather than a
    constraint violation midway through it.
    """
    lines: list[str] = [""]

    if files:
        lines.append(f"SQL files ({len(files)}) — run in this order:")
        for path in files[:_PLAN_MAX_FILES]:
            name = path.name if len(path.name) <= 29 else path.name[:28] + "…"
            lines.append(f"  {name:<30}{_human_size(path.stat().st_size):>10}")
        if len(files) > _PLAN_MAX_FILES:
            lines.append(f"  … and {len(files) - _PLAN_MAX_FILES} more")
    else:
        lines.append("SQL files : none  (SQL_DIR is None)")

    lines.append("")
    if csv_mapping:
        lines.append(f"CSV loads : {_plural(len(csv_mapping), 'file')} → tables")
        # A missing CSV is skipped with a warning mid-run, which is easy to
        # scroll past and leaves a table silently empty. Say so up front,
        # while declining is still free.
        for name in missing_csvs:
            lines.append(f"  ⚠ MISSING: {name[:30]}")
    else:
        lines.append("CSV loads : none  (CSV_MAPPING is empty)")
    return lines


def confirm_reset(
    db_name: str,
    config_init: dict[str, str],
    recreate: bool,
    files: list[Path],
    csv_mapping: dict[str, str],
    missing_csvs: list[str],
) -> None:
    """
    Describe exactly what is about to happen and wait for confirmation.

    Rollback guarantee: this runs BEFORE any destructive operation, so
    answering 'no' leaves the database completely untouched — there is
    nothing to roll back.
    """
    exists = _database_exists(db_name, config_init)
    plan = _plan_lines(files, csv_mapping, missing_csvs)

    if recreate and exists:
        _print_box(
            "⚠  WARNING — DESTRUCTIVE OPERATION",
            [
                "",
                f"Database : {db_name}",
                "",
                "This database WILL BE PERMANENTLY DELETED and",
                "rebuilt from scratch. ALL existing data is lost.",
                *plan,
            ],
            YELLOW,
        )
    elif exists:
        _print_box(
            "ℹ  APPLY TO EXISTING DATABASE",
            [
                "",
                f"Database : {db_name}",
                "",
                "The database is kept. Everything below runs on",
                "top of what is already there.",
                *plan,
            ],
            CYAN,
        )
    else:
        _print_box(
            "ℹ  FIRST-TIME SETUP",
            [
                "",
                f"Database : {db_name}",
                "",
                "No existing database was found.",
                "A fresh database will be created.",
                *plan,
            ],
            CYAN,
        )

    print()

    try:
        answer = input(_c("  Proceed? [yes / no]  → ", BOLD)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment or Ctrl-C — treat as 'no'
        answer = "no"

    print()

    if answer in ("yes", "y"):
        print(_c("  ✔  Confirmed. Starting setup...", BOLD, GREEN))
        print()
    else:
        # Nothing has been modified yet, so exiting here is a complete and
        # safe rollback of the entire operation.
        print(_c("  ✖  Aborted. No changes were made.", BOLD, RED))
        print()
        sys.exit(0)


def drop_database(db_name: str, config_init: dict[str, str]) -> None:
    """Drop the database if it exists, disconnecting any active sessions."""
    with Spinner(f"Dropping database '{db_name}'"):
        conn = psycopg2.connect(**config_init)
        conn.autocommit = True
        cursor = conn.cursor()
        try:
            # Terminate active connections first — without this, DROP
            # DATABASE fails whenever any session is still connected.
            cursor.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                [db_name],
            )
            cursor.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name))
            )
        finally:
            cursor.close()
            conn.close()


def create_database(db_name: str, config_init: dict[str, str]) -> None:
    """Create the database, leaving it alone if it already exists."""
    with Spinner(f"Creating database '{db_name}'"):
        conn = psycopg2.connect(**config_init)
        conn.autocommit = True
        cursor = conn.cursor()
        try:
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
            )
        except psycopg2.errors.DuplicateDatabase:
            pass   # already exists — spinner will still show ✅
        finally:
            cursor.close()
            conn.close()


# ============================================================
# SQL FILES
# ============================================================

def sql_files(sql_dir: Path) -> list[Path]:
    """
    Return the top-level .sql files in sql_dir, sorted by filename.

    The glob is deliberately non-recursive. Sibling sub-directories hold SQL
    belonging to a later stage of the pipeline (staging, marts, tests), which
    must not be swept into the bootstrap — so a project points SQL_DIR at the
    exact folder it wants run, e.g. sql/oltp rather than sql.
    """
    if sql_dir.is_file():
        raise NotADirectoryError(
            f"SQL_DIR must be a directory of .sql files, but points at a "
            f"single file:\n    {sql_dir}\n  Use its folder instead:\n"
            f"    {sql_dir.parent}"
        )
    if not sql_dir.is_dir():
        raise FileNotFoundError(f"SQL_DIR not found: {sql_dir}")

    files = sorted(sql_dir.glob("*.sql"))
    if not files:
        # A folder of folders is the normal layout for a multi-model project
        # (sql/oltp, sql/marts), and pointing one level too high is the
        # obvious mistake — so name the candidates rather than just refusing.
        nested = sorted({path.parent for path in sql_dir.glob("*/*.sql")})
        hint = ""
        if nested:
            candidates = "\n".join(f"      {path}" for path in nested)
            hint = (
                f"\n  These sub-folders do contain .sql files — point SQL_DIR "
                f"at the one to run:\n{candidates}"
            )
        raise FileNotFoundError(
            f"No .sql files directly inside {sql_dir} "
            f"(sub-directories are not searched){hint}"
        )
    return files


# A COPY whose rows sit inline in the file, which is what pg_dump writes by
# default. Postgres streams that payload over a separate channel of the
# client protocol, so the statement cannot be sent through execute() at all.
# [^;]* is bounded by the statement terminator, so a match cannot run past
# the COPY it belongs to.
_COPY_FROM_STDIN = re.compile(
    r"^\s*COPY\b[^;]*\bFROM\s+stdin\b", re.IGNORECASE | re.MULTILINE
)

# Backslash commands are features of the psql client, interpreted before
# anything reaches the server, so no driver can run them.
#
# Matching any \word would be simpler but reads a raw file with no lexical
# context, so a line beginning with a backslash inside a string literal or a
# dollar-quoted body (\section{...} in a text value, say) would condemn a
# perfectly valid file. Restricting it to psql's actual vocabulary keeps that
# to collisions with a real command name. Longest alternative first, so \c
# cannot shadow \copy, and derived from the tuple rather than hand-ordered.
_PSQL_COMMANDS = (
    "copy", "i", "ir", "include", "include_relative", "set", "unset", "echo",
    "qecho", "warn", "c", "connect", "conninfo", "cd", "setenv", "timing",
    "pset", "o", "out", "g", "gexec", "gset", "gdesc", "watch", "crosstabview",
    "encoding", "password", "edit", "ef", "ev", "sf", "sv", "lo_import",
    "lo_export", "lo_list", "lo_unlink", "if", "elif", "else", "endif", "q",
)
#
# psql's \d family (\d, \dt, \df) is deliberately NOT matched. Those only
# print a listing to the console, so they are near-worthless in a bootstrap
# file, while '\d' is everyday regex — a dollar-quoted body wrapping a line
# onto a leading \d+ is far likelier than someone seeding data with \dt.
_PSQL_META = re.compile(
    r"^[ \t]*\\(" + "|".join(sorted(_PSQL_COMMANDS, key=len, reverse=True))
    + r")(?![a-zA-Z0-9_])",
    re.MULTILINE,
)


def _line_number(text: str, offset: int) -> int:
    """Return the 1-based line containing a character offset."""
    return text.count("\n", 0, offset) + 1


def _read_sql(path: Path) -> str:
    """
    Read one .sql file, ready to execute.

    Decoded as utf-8-sig rather than utf-8: files saved by Windows editors
    often begin with a byte-order mark, and Postgres reports that as a syntax
    error at or near "" with nothing pointing at the real cause. The codec
    strips a BOM when present and is identical to utf-8 when it is not.

    The two guards below catch files that are valid input for the psql
    command-line client but were never valid input for a database driver.
    Both otherwise fail as a bare syntax error pointing at the wrong thing.
    """
    text = path.read_text(encoding="utf-8-sig")

    if _COPY_FROM_STDIN.search(text):
        raise ValueError(
            f"{path.name} holds a 'COPY ... FROM stdin' block with its rows "
            f"inline — the format pg_dump writes by default. Postgres streams "
            f"those rows over a separate channel, so the file cannot run as "
            f"ordinary SQL. Re-export it with 'pg_dump --inserts', or change "
            f"the statement to COPY ... FROM '/server/side/path.csv', or "
            f"restore that particular file with psql."
        )

    meta = _PSQL_META.search(text)
    if meta:
        line = _line_number(text, meta.start())
        raise ValueError(
            f"{path.name}:{line} uses the psql meta-command '\\{meta.group(1)}'. "
            f"psql interprets those itself — they never reach the server, so no "
            f"driver can run them. Equivalents here: '\\copy' → load the file "
            f"through CSV_MAPPING, or use SQL COPY with a path the server can "
            f"read; '\\i' → drop the included file into SQL_DIR and let the "
            f"filename order run it."
        )
    return text


_EXCERPT_HALF = 46


def _excerpt(line: str, column: int) -> tuple[str, str]:
    """Return a window of a long line plus a caret marking the column."""
    start = max(column - 1 - _EXCERPT_HALF, 0)
    end = min(start + 2 * _EXCERPT_HALF, len(line))
    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(line) else ""
    # Tabs would desynchronise the caret from the text above it.
    window = line[start:end].replace("\t", " ")
    caret = " " * (len(prefix) + (column - 1 - start)) + "^"
    return prefix + window + suffix, caret


def _report_sql_error(path: Path, text: str, exc: BaseException) -> None:
    """
    Say exactly where in the file the failure happened.

    Because each file is sent as one batch, the character offset Postgres
    reports is an offset into that file — so it can be resolved back to a
    line, a column and the offending source line. Without this, a failure in
    a 4,000-line data file only says 'syntax error at or near ...'.
    """
    if not isinstance(exc, psycopg2.Error) or exc.diag is None:
        print(_c(f"     → {exc}", RED))
        return

    diag = exc.diag
    print(_c(f"     → {diag.message_primary or exc}", RED))

    position = diag.statement_position
    if position is not None and text:
        offset = max(int(position) - 1, 0)
        line_no = _line_number(text, offset)
        line_start = text.rfind("\n", 0, offset) + 1
        line_end = text.find("\n", offset)
        line = text[line_start:] if line_end == -1 else text[line_start:line_end]
        source, caret = _excerpt(line, offset - line_start + 1)
        print(_c(f"     → {path}:{line_no}:{offset - line_start + 1}", YELLOW))
        print(_c(f"         {source}", DIM))
        print(_c(f"         {caret}", YELLOW))

    for label, value in (
        ("DETAIL ", diag.message_detail),
        ("HINT   ", diag.message_hint),
        ("CONTEXT", diag.context),
    ):
        if value:
            print(_c(f"     → {label}  {value.strip()}", DIM))


def run_sql_files(files: list[Path], config: dict[str, str]) -> None:
    """
    Execute each file as a single batch, in order, exactly as written.

    Running the files as written is what guarantees that everything in them
    actually executes. Extracting statements with a regex silently drops
    whatever the pattern does not match (CREATE INDEX, CREATE VIEW,
    dollar-quoted function bodies, schema-qualified names) with no error to
    reveal the omission.

    Each file is committed on its own, so a failure leaves the files before
    it applied. That is deliberate: a single wrapping transaction would be
    tidier but would forbid the statements that cannot run inside one
    (CREATE INDEX CONCURRENTLY, VACUUM, some ALTER TYPE forms). The failure
    report below names the file that stopped, which is what makes the
    partial state recoverable — fix that file and re-run.
    """
    conn = psycopg2.connect(**config)
    cursor = conn.cursor()
    try:
        for path in files:
            spinner = Spinner(f"Running  {_c(f'{path.name:<26}', BOLD, WHITE)}")
            spinner.start()
            started = time.perf_counter()
            text = ""
            try:
                text = _read_sql(path)
                # No parameters are passed, so psycopg2 performs no
                # interpolation and a literal '%' in the SQL is safe.
                cursor.execute(text)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                spinner.stop(success=False, suffix="failed")
                _report_sql_error(path, text, exc)
                raise
            spinner.stop(
                success=True, suffix=f"{time.perf_counter() - started:.2f}s"
            )
    finally:
        cursor.close()
        conn.close()


# ============================================================
# DATA LOADING
# ============================================================

def _quote_literal(value: str) -> str:
    """Render a Python string as a single-quoted SQL literal."""
    return "'" + value.replace("'", "''") + "'"


def _quote_columns(columns: list[str]) -> str:
    """Render column names as a quoted, comma-separated identifier list."""
    return ", ".join('"' + col.strip().replace('"', '""') + '"' for col in columns)


def _load_stream(
    csv_path: Path,
    table_name: str,
    cursor: extensions.cursor,
    dialect: dict[str, str],
) -> int:
    """
    Pipe the CSV file straight into Postgres and let Postgres parse it.

    Nothing passes through pandas, so values reach the table exactly as
    written in the file and memory use stays flat regardless of file size.
    This also sidesteps the float-upcast problem described in
    _restore_integer_columns entirely — it simply cannot arise here.

    Column names are taken from the CSV header, so the file's column order
    does not have to match the table's.
    """
    with csv_path.open("r", encoding=dialect["encoding"], newline="") as handle:
        header = next(csv.reader([handle.readline()], delimiter=dialect["delimiter"]))
        handle.seek(0)   # rewind so COPY's HEADER option consumes the header
        # table_name comes from CSV_MAPPING (trusted config) and may be
        # schema-qualified, so it is inlined rather than quoted as one
        # identifier — sql.Identifier would turn raw.sales into "raw.sales".
        cursor.copy_expert(
            f"COPY {table_name} ({_quote_columns(header)}) FROM STDIN WITH ("
            f"FORMAT csv, HEADER true, "
            f"DELIMITER {_quote_literal(dialect['delimiter'])}, "
            f"NULL {_quote_literal(dialect['null_token'])})",
            handle,
        )
    return cursor.rowcount


def _restore_integer_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Undo pandas' float upcast on integer columns that contain nulls.

    pandas' int64 dtype cannot represent a missing value, so read_csv has no
    choice but to store any integer column containing nulls as float64 —
    186 silently becomes 186.0, and Postgres then rejects the text "186.0"
    for an integer column.

    Where every non-null value in a float column is whole, the column was an
    integer to begin with and is restored to the nullable Int64 dtype.
    Columns holding genuine decimals fail that test, stay float, and keep
    full precision — which is why this is preferred over blanket-rounding
    every float with to_csv(float_format='%.0f').
    """
    for column in frame.select_dtypes(include=["float32", "float64"]).columns:
        non_null = frame[column].dropna()
        if not non_null.empty and (non_null % 1 == 0).all():
            frame[column] = frame[column].astype("Int64")
    return frame


def _load_pandas(
    csv_path: Path,
    table_name: str,
    cursor: extensions.cursor,
    dialect: dict[str, str],
) -> int:
    """
    Read the CSV into a DataFrame, then COPY that in as CSV text.

    Only worth using when the data has to be touched in Python before it
    lands; otherwise prefer _load_stream. CSV is used as the wire format
    (rather than tab-separated text) because pandas quotes embedded
    delimiters, quotes and newlines correctly for it — a tab-separated
    export would corrupt any value containing a tab or a backslash.

    NULLs travel as an unquoted \\N, which CSV format treats as NULL only
    when unquoted, so genuine empty strings survive as empty strings.
    """
    # Imported here, not at module scope: a pure-SQL project has no CSVs to
    # load and should not need pandas installed to bootstrap its database.
    import pandas as pd

    na_values = [dialect["null_token"]] if dialect["null_token"] else None
    frame = pd.read_csv(
        csv_path,
        delimiter=dialect["delimiter"],
        encoding=dialect["encoding"],
        na_values=na_values,
    )
    frame = _restore_integer_columns(frame)
    buffer = io.StringIO(
        frame.to_csv(index=False, header=False, na_rep="\\N")
    )
    cursor.copy_expert(
        f"COPY {table_name} ({_quote_columns(list(frame.columns))}) "
        f"FROM STDIN WITH (FORMAT csv, NULL '\\N')",
        buffer,
    )
    return len(frame)


_LOADERS = {"stream": _load_stream, "pandas": _load_pandas}


def _validate_load_mode(load_mode: str) -> None:
    """Reject an unknown LOAD_MODE. Called at plan time, before the drop."""
    if load_mode not in _LOADERS:
        raise ValueError(
            f"LOAD_MODE must be one of {sorted(_LOADERS)}, got {load_mode!r}"
        )


def load_csv_to_table(
    csv_path: Path,
    table_name: str,
    config: dict[str, str],
    load_mode: str,
    dialect: dict[str, str],
) -> None:
    """Load one CSV into one table, with a live spinner and a row count."""
    _validate_load_mode(load_mode)

    if not csv_path.exists():
        print(f"  {_c('⚠', YELLOW)}  Not found — skipping: {_c(csv_path, DIM)}")
        return
    if csv_path.stat().st_size == 0:
        print(f"  {_c('⚠', YELLOW)}  Empty — skipping: {_c(csv_path.name, DIM)}")
        return

    conn = psycopg2.connect(**config)
    cursor = conn.cursor()
    spinner = Spinner(f"Loading  {_c(f'{table_name:<22}', BOLD, WHITE)}").start()

    try:
        rows = _LOADERS[load_mode](csv_path, table_name, cursor, dialect)
        conn.commit()
        spinner.stop(success=True, suffix=f"{rows:,} rows")
    except Exception as exc:
        conn.rollback()
        spinner.stop(success=False, suffix="failed")
        print(_c(f"     → {exc}", RED))
        raise
    finally:
        cursor.close()
        conn.close()


def load_all_csvs(
    csv_mapping: dict[str, str],
    config: dict[str, str],
    load_mode: str,
    dialect: dict[str, str],
) -> None:
    """Load every CSV in the mapping, in the order it is declared."""
    for rel_path, table_name in csv_mapping.items():
        load_csv_to_table(_ROOT / rel_path, table_name, config, load_mode, dialect)


# ============================================================
# VERIFICATION
# ============================================================

# This report answers "what did my .sql files build?", so it must exclude
# everything the project did not create. Three filters do that, and each one
# is load-bearing:
#   - the ^pg_ test is not redundant with the NOT IN: pg_toast holds an index
#     per table, and without it those drown the report;
#   - partitions are skipped so their rows are not counted twice, being
#     already included in the partitioned parent's count;
#   - pg_depend deptype 'e' marks an object as owned by an EXTENSION. Those
#     live in ordinary schemas — CREATE EXTENSION postgis puts a populated
#     spatial_ref_sys table and ~1,000 functions straight into public — so
#     without this the report describes the extensions, not the project.
# Both queries apply the same three, so the table list and the object counts
# below it agree about what "belongs to this project" means.
_NOT_FROM_EXTENSION = """
      AND NOT EXISTS (
          SELECT 1 FROM pg_depend d
          WHERE d.objid = {oid} AND d.classid = '{catalog}'::regclass
            AND d.deptype = 'e'
      )
"""

# relkind: r = table, p = partitioned table, v = view,
#          m = materialized view, i = index, S = sequence.
_RELATIONS_SQL = f"""
    SELECT c.relkind, n.nspname, c.relname
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
      AND n.nspname !~ '^pg_'
      AND NOT c.relispartition
      AND c.relkind IN ('r', 'p', 'v', 'm', 'i', 'S')
      {_NOT_FROM_EXTENSION.format(oid="c.oid", catalog="pg_class")}
    ORDER BY 2, 3
"""

_ROUTINES_SQL = f"""
    SELECT count(*)
    FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
      AND n.nspname !~ '^pg_'
      {_NOT_FROM_EXTENSION.format(oid="p.oid", catalog="pg_proc")}
"""


def verify_database(config: dict[str, str]) -> tuple[int, int]:
    """
    Report what the database actually contains, read back from the catalog.
    Returns (number of tables, total rows across them).

    This is the only check that covers both load paths. A multi-statement SQL
    batch reports only its last statement's rowcount, so when the data ships
    inside .sql files there is nothing client-side to trust — the rows have to
    be counted in the database. COUNT(*) is exact, unlike the planner's
    reltuples estimate, which sits at 0 until something runs ANALYZE.

    Objects other than tables are summarised too, because a pure-SQL project
    builds views, functions and indexes from those same files and a silent
    zero there is exactly the kind of failure worth seeing at the end of a run.

    The totals are returned rather than judged here: an empty table is not an
    error on its own — a schema-only bootstrap ends with every table empty and
    is perfectly correct — so the caller, which knows whether data was even
    supposed to load, decides what to say about it.
    """
    conn = psycopg2.connect(**config)
    cursor = conn.cursor()
    try:
        cursor.execute(_RELATIONS_SQL)
        relations: list[tuple[str, str, str]] = cursor.fetchall()
        cursor.execute(_ROUTINES_SQL)
        routines: int = cursor.fetchone()[0]

        tables = [(s, r) for kind, s, r in relations if kind in ("r", "p")]
        if not tables:
            print(f"  {_c('⚠', YELLOW)}   No tables exist in this database.")

        total_rows = 0
        for schema, table in tables:
            cursor.execute(
                sql.SQL("SELECT count(*) FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            rows: int = cursor.fetchone()[0]
            total_rows += rows
            icon = _c("✅", GREEN) if rows else _c("⚠ ", YELLOW)
            print(f"  {icon}  {f'{schema}.{table}':<40}{rows:>13,} rows")

        kinds = Counter(kind for kind, _, _ in relations)
        extras = [
            _plural(count, singular, plural)
            for count, singular, plural in (
                (kinds.get("v", 0), "view", ""),
                (kinds.get("m", 0), "materialized view", ""),
                (routines, "function", ""),
                (kinds.get("i", 0), "index", "indexes"),
                (kinds.get("S", 0), "sequence", ""),
            )
            if count
        ]
        if extras:
            print()
            print(_c(f"  Also present:  {'  ·  '.join(extras)}", DIM))
        return len(tables), total_rows
    finally:
        cursor.close()
        conn.close()


# ============================================================
# ENTRY POINT
# ============================================================

class PlanError(Exception):
    """
    A CONFIG or on-disk problem caught before the database was touched.

    Carried as its own type so the exit path can state that fact — the
    difference between 'nothing was touched' and 'the database is half-built'
    is the first thing anyone needs to know after a failure.
    """


def plan_setup() -> tuple[list[Path], list[str]]:
    """
    Resolve and check everything knowable without touching the database.

    Every mistake this catches is a mistake in CONFIG or in a file on disk,
    which means all of them are knowable before the run starts — and a run
    that starts by dropping the database must not discover them afterwards.
    Failing here leaves the existing database untouched; failing after the
    drop would destroy it and then refuse to build the replacement.

    Returns the .sql files to run and the names of any missing CSVs.
    """
    files = sql_files(SQL_DIR) if SQL_DIR is not None else []

    # Reading every file twice (here, then again to execute) is the cost of
    # knowing before the drop that all of them can run. Each is released
    # before the next is read, so memory stays flat however large the folder.
    for path in files:
        _read_sql(path)

    if CSV_MAPPING:
        _validate_load_mode(LOAD_MODE)

    missing = [name for name in CSV_MAPPING if not (_ROOT / name).exists()]
    return files, missing


def _report_partial_state(db_name: str) -> None:
    """
    Describe what survived a failed run.

    Each .sql file commits on its own, so a failure part-way through leaves
    the files before it applied. Saying nothing — or worse, claiming nothing
    ran — would misrepresent a database that is now half-built.
    """
    print()
    _print_box(
        "⚠  PARTIAL STATE — the run stopped part-way",
        [
            "",
            f"Database : {db_name}",
            "",
            "Steps that completed above are committed and are",
            "still in the database. Fix the cause and re-run;",
            "with RECREATE_DB = True the re-run starts clean.",
        ],
        YELLOW,
    )
    try:
        print()
        verify_database(DB_CONFIG)
    except Exception as exc:   # noqa: BLE001 — a best-effort diagnostic only
        print(_c(f"  (could not read the database back: {exc})", DIM))


def run_setup() -> None:
    """Create the database, run the SQL files, load any CSVs, then verify."""
    _print_header("DATABASE SETUP")
    db_name = DB_CONFIG["dbname"]

    try:
        files, missing_csvs = plan_setup()
    except Exception as exc:
        raise PlanError(str(exc)) from exc

    # Guard: describe the operation and require explicit confirmation. This
    # runs before anything destructive, so 'no' guarantees zero changes.
    confirm_reset(db_name, DB_CONFIG_INIT, RECREATE_DB, files, CSV_MAPPING,
                  missing_csvs)

    step = itertools.count(1)

    try:
        if RECREATE_DB:
            _print_section(f"Step {next(step)} — Drop existing database")
            drop_database(db_name, DB_CONFIG_INIT)

        _print_section(f"Step {next(step)} — Create database")
        create_database(db_name, DB_CONFIG_INIT)

        if files:
            _print_section(f"Step {next(step)} — Run SQL files ({len(files)})")
            print()
            run_sql_files(files, DB_CONFIG)

        if CSV_MAPPING:
            _print_section(f"Step {next(step)} — Load CSVs ({LOAD_MODE} mode)")
            print()
            load_all_csvs(CSV_MAPPING, DB_CONFIG, LOAD_MODE, CSV_DIALECT)
    except Exception:
        _report_partial_state(db_name)
        raise

    _print_section(f"Step {next(step)} — Verify")
    print()
    n_tables, total_rows = verify_database(DB_CONFIG)

    # State the outcome as measured, not as intended. An empty database is
    # not necessarily a failed one — a schema-only bootstrap is supposed to
    # end this way — so the wording changes while the exit status does not.
    print()
    if n_tables == 0:
        _print_box("⚠  Finished — but the database has no tables.", [], YELLOW)
    elif total_rows == 0:
        _print_box("✔  Setup complete — tables built, 0 rows loaded.", [], CYAN)
    else:
        _print_box("✔  Setup complete — database is ready.", [], GREEN)
    print()


if __name__ == "__main__":
    try:
        run_setup()
    except Exception as exc:
        # A config mistake (bad SQL_DIR) or a failing .sql file has already
        # been described in detail above; a traceback would only bury it.
        planning = isinstance(exc, PlanError)
        cause = exc.__cause__ if planning and exc.__cause__ else exc
        print()
        print(_c(f"  {type(cause).__name__}: {cause}", RED))
        print()
        _print_box(
            "✖  Setup failed — nothing was touched." if planning
            else "✖  Setup failed — see the state report above.",
            [],
            RED,
        )
        print()
        sys.exit(1)
