# Media Condenser (mcon)

Scales images and videos down in a directory tree to save disk space, preserving metadata.
It focusses on simplicity, reasonable defaults, good ux and the option to be used in scripts.

- Videos → 720p H.265, constant quality 23, audio untouched
- Images → 2048 px on the longest side
- Motion photos → primary, Ultra HDR gain map and embedded video handled as one container
  - This feature is experimental
- Anything already within target is **skipped**, byte-for-byte
- Anything of a type it cannot process (HEIC, AVIF, raw) is reported as unsupported and left alone

Requires `ffmpeg`, `ffprobe`, `exiftool` and `magick` on `PATH`.

## Motivation

Archiving old photos and videos can take up a lot of space and most of this data is basically wasted.
4k videos from a phone with inefficient encoding, enourmous photos which are basically good how they are without any need to crop/edit them ever again.
It's nice to have an ultra high resolution image or video, but is it really necessary? Slightly scaling down these media files from phones aren't even noticeable, because of sensor noise, motion blur, focus blur, lens distortion, etc, but can result in a huge disk space saving.
But selecting the right parameters can be hard and running controlling ffmpeg to do all that can quickly become tricky if you want to do it right.
So I created this tool, to control ffmpeg and more instead. You just give it a directory and a target (or just replace the data) and this tools does everything automatically.

## Usage

```bash
# See what would happen. Writes nothing.
uv run mcon --dry-run ~/Photos --output-dir ~/Photos_compressed

# Full run, output into a mirror tree
uv run mcon ~/Photos --output-dir ~/Photos_compressed

# Overwrite originals (temp file + atomic rename per file)
uv run mcon ~/Photos --strategy replace
```

| Option                         | Purpose                                                                                                                                                                                                                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dry-run` / `-n`             | Classify everything, print the plan, write nothing. Also checks the output tree, so a rerun preview reports the files it would skip as already present                                                                                                                                    |
| `--verify` / `--no-verify`     | **On by default.** Checks each output _before_ committing it — re-extracts components, confirms they decode, confirms metadata survived. Anything that fails is discarded and reported as failed, leaving the original untouched. `--no-verify` skips the checks and the probes they cost |
| `--strategy`                   | `copy` (default, never touches sources; requires `--output-dir`) or `replace`                                                                                                                                                                                                             |
| `--output-dir` / `-o`          | Mirror output tree; skipped files are copied across unchanged so the tree is complete. **Required by `copy`** — a copy run with nowhere to write is refused rather than guessed at. Refused with `--strategy replace`, which writes in place and would ignore it                          |
| `--jobs N` / `-j`              | Concurrent encodes (default 4; encoder thread pools are derived so jobs × pools ≈ cores)                                                                                                                                                                                                  |
| `--verbose` / `-v`             | Debug logs, including every ffmpeg/exiftool command line. `-vv` adds third-party libraries                                                                                                                                                                                                |
| `--quiet` / `-q`               | Warnings and errors only, and no progress bar                                                                                                                                                                                                                                             |
| `--log-level`                  | Explicit level. Wins over `-v` and `-q`                                                                                                                                                                                                                                                   |
| `--log-format`                 | `rich`, `plain` or `json`. Defaults to `rich` on a terminal and `plain` when redirected                                                                                                                                                                                                   |
| `--progress` / `--no-progress` | Force the live progress bar on or off. Unset, it appears when stderr is a terminal showing rich logs                                                                                                                                                                                      |
| `--summary`                    | `table` (default), `json` for one document on stdout, or `none`                                                                                                                                                                                                                           |
| `--report-json PATH`           | Write the per-file results as JSON to a file, for scripting                                                                                                                                                                                                                               |

## Configuration

Global config at `~/.config/mcon/config.yaml`:

```yaml
strategy: copy
jobs: 4
rules:
  images:
    max_edge: 2048
    quality: 85
    skip_png: true
  videos:
    max_short_edge: 720
    crf: 23
```

A full example config is located at `example-config.yaml`

Drop a `.mcon.yaml` in any media directory to override rules for that directory and
everything below it:

```yaml
# ~/Photos/Scans/.mcon.yaml — keep scans at full resolution
rules:
  images:
    enabled: false
```

## Output

Results and progress go to **different streams**, so both piping and silencing work:

```bash
mcon ~/Photos -o ~/out > summary.txt   # the plan and summary tables (stdout)
mcon ~/Photos -o ~/out 2>/dev/null     # drop the bar and the log lines (stderr)
```

On a terminal you get a byte-weighted progress bar pinned to the bottom — weighted by
bytes because a 4 GB video and a 2 MB JPEG are not comparable work, and a file-count
ETA on a mixed library is wrong for most of the run — with one row per file currently
being encoded, and log lines scrolling above it.

Redirect it and the tool switches to plain timestamped lines with no bar.

The two streams are formatted independently, which is what makes batch use
comfortable — machine-readable results with human-readable logs, or results with no
logs at all:

```bash
# One JSON document on stdout instead of the tables
mcon ~/Photos -o ~/out --summary json | jq '.totals.compression_rate'

# Results only, no logs
mcon ~/Photos -o ~/out --summary json 2>/dev/null > run.json

# Exit code only; nothing on stdout at all
mcon ~/Photos -o ~/out --summary none -q

# Tables on screen and a JSON file on the side
mcon ~/Photos -o ~/out --report-json run.json

# One JSON object per log record, e.g. to pick out failures as they happen
mcon ~/Photos -o ~/out --log-format json 2>&1 >/dev/null | jq -c 'select(.outcome == "failed")'
```

`--summary json` covers `--dry-run` too, so the plan is machine-readable as well.

Every result is written after the final log line, so the per-file verification lines
cannot scroll the summary off the screen.

Exit codes: `0` clean, `1` a file failed or failed verification, `2` a config
error, `130` interrupted with Ctrl-C (in-flight temp files are cleaned up first, and
whatever finished is still reported).

## Logging

Logging is a [`logging.config.dictConfig`][dictconfig] document. The packaged default
is `src/media_condenser/default_logging.yaml`; anything under `logging:` in your config file is
merged onto it **per leaf key**, so a fragment keeps every formatter and handler it
does not mention:

```yaml
logging:
  loggers:
    media_condenser:
      level: INFO
    media_condenser.probe:
      level: DEBUG # just the ffprobe/exiftool command lines
```

Useful logger names: `media_condenser.pipeline` (per-file outcomes), `media_condenser.handlers.video` and
`media_condenser.probe` (subprocess command lines and their full stderr, at DEBUG),
`media_condenser.storage`, `media_condenser.discovery`.

Because the document is yours to extend, **a log file needs no dedicated flag** — add
a handler and name it. `--log-format` only ever swaps the `rich`/`plain`/`json`
handler, so your own handlers stay wired up under any format:

```yaml
logging:
  handlers:
    file:
      class: logging.handlers.RotatingFileHandler
      filename: /home/you/.local/state/mcon.log
      maxBytes: 1048576
      backupCount: 3
      formatter: plain
      level: DEBUG
  loggers:
    media_condenser:
      level: DEBUG
      handlers: [rich, file]
```

[dictconfig]: https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
