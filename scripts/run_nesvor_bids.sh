#!/usr/bin/env bash
#
# Run NeSVoR reconstruction over every subject/session in a BIDS folder.
#
# Usage:
#   ./run_nesvor_bids.sh [-n|--dry-run] [-m|--masks <masks_bids_dir>] \
#       <input_bids_dir> <output_bids_dir> [log_file]
#
# Options:
#   -n, --dry-run        Print the docker command for each subject/session
#                        instead of running it (no containers, no files written).
#   -m, --masks <dir>    BIDS folder of stack masks (e.g. produced by
#                        make_bids_masks.py). For each T2w stack, the matching
#                        *_T2w_mask.nii.gz at the same relative path is passed to
#                        NeSVoR via --stack-masks.
#
# Example:
#   ./run_nesvor_bids.sh \
#       --masks /data/mybids/derivatives/masks \
#       /data/mybids/derivatives/masked \
#       /data/mybids/derivatives/nesvor
#
# Each subject/session is reconstructed from all its T2w stacks in <sub>/<ses>/anat/.
# The output is written in BIDS layout, with the `run-N` entity replaced by
# `srr-nesvor`, e.g.:
#   sub-001/ses-01/anat/sub-001_ses-01_acq-haste_srr-nesvor_T2w.nii.gz
#
# Alongside the volume, three extra NeSVoR outputs are saved under
# <output_bids_dir>/derivatives/nesvor_extras/<sub>/<ses>/anat/ :
#   ..._srr-nesvor_slices/      motion-corrected (acquired) slices  (--output-slices)
#   ..._srr-nesvor_simslices/   simulated slices from the volume    (--simulated-slices)
#   ..._srr-nesvor_model.pt     the trained NeSVoR model            (--output-model)
#
# NeSVoR runs inside the official Docker image. The input BIDS dir is mounted
# read-only at /incoming, the masks dir (if given) read-only at /masks, and the
# output dir read-write at /outgoing; all host paths are translated to these
# container mount points.
#
# All NeSVoR logs/output are redirected to the log file; the terminal only shows
# a tqdm-style progress bar.

set -u -o pipefail

# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
MASKS_DIR=""

# Parse optional flags (may appear before the positional arguments).
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1; shift ;;
        -m|--masks)   MASKS_DIR="${2%/}"; shift 2 ;;
        --masks=*)    MASKS_DIR="${1#*=}"; MASKS_DIR="${MASKS_DIR%/}"; shift ;;
        --) shift; POSITIONAL+=("$@"); break ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)  POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [-n|--dry-run] [-m|--masks <masks_bids_dir>] <input_bids_dir> <output_bids_dir> [log_file]" >&2
    exit 1
fi

INPUT_DIR="${1%/}"
OUTPUT_DIR="${2%/}"
LOG_FILE="${3:-${OUTPUT_DIR}/nesvor_$(date +%Y%m%d_%H%M%S).log}"

# NeSVoR options (tweak as needed)
OUTPUT_RESOLUTION="1.0"
BATCH_SIZE="4096"
N_LEVELS_BIAS="1"

# Docker settings
DOCKER_IMAGE="${NESVOR_IMAGE:-junshenxu/nesvor:v0.5.0}"
CONTAINER_IN="/incoming"    # input dir mount point (read-only) inside container
CONTAINER_MASKS="/masks"    # masks dir mount point (read-only) inside container
CONTAINER_OUT="/outgoing"   # output dir mount point (read-write) inside container

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

if [[ -n "$MASKS_DIR" && ! -d "$MASKS_DIR" ]]; then
    echo "Error: masks directory does not exist: $MASKS_DIR" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: 'docker' command not found in PATH." >&2
    exit 1
fi

# Resolve to absolute paths so the bind mounts are unambiguous.
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
[[ -n "$MASKS_DIR" ]] && MASKS_DIR="$(cd "$MASKS_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# Map a stack filename to its mask filename (mirrors make_bids_masks.py):
#   ..._T2w<anything>.nii.gz -> ..._T2w_mask.nii.gz
#   anything else            -> insert _mask before the extension.
mask_name_for() {
    local base="$1"
    if [[ "$base" == *_T2w*.nii.gz ]]; then
        printf '%s_T2w_mask.nii.gz' "${base%%_T2w*}"
    else
        printf '%s_mask.nii.gz' "${base%.nii.gz}"
    fi
}

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

# ----------------------------------------------------------------------------
# Collect tasks: every anat directory that contains T2w stacks
# Only descend into top-level BIDS subject folders (sub-*); ignore anything
# else next to them (code/, derivatives/, sourcedata/, ...).
# ----------------------------------------------------------------------------
ANAT_DIRS=()
for sub_dir in "$INPUT_DIR"/sub-*/; do
    [[ -d "$sub_dir" ]] || continue   # no sub-* match -> glob stays literal
    while IFS= read -r d; do
        ANAT_DIRS+=("$d")
    done < <(find "$sub_dir" -type d -name anat 2>/dev/null | sort)
done

if [[ ${#ANAT_DIRS[@]} -eq 0 ]]; then
    echo "No sub-*/.../anat folders found under: $INPUT_DIR" >&2
    exit 1
fi

TASKS=()
for anat in "${ANAT_DIRS[@]}"; do
    # Only keep anat dirs that actually contain T2w nifti stacks
    if compgen -G "${anat}/*T2w*.nii.gz" >/dev/null; then
        TASKS+=("$anat")
    fi
done

TOTAL=${#TASKS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No anat folders with *T2w*.nii.gz stacks found under: $INPUT_DIR" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# tqdm-style progress bar
# ----------------------------------------------------------------------------
START_TIME=$(date +%s)

draw_bar() {
    local current=$1 total=$2 label=$3
    local width=30
    local frac=0
    (( total > 0 )) && frac=$(( current * 100 / total ))
    local filled=$(( current * width / total ))
    (( filled > width )) && filled=$width
    local bar
    bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
    bar+=$(printf '%*s' "$(( width - filled ))" '' | tr ' ' '-')

    # elapsed / eta
    local now elapsed eta="--:--"
    now=$(date +%s)
    elapsed=$(( now - START_TIME ))
    if (( current > 0 )); then
        local per=$(( elapsed / current ))
        local remaining=$(( per * (total - current) ))
        eta=$(printf '%02d:%02d' $(( remaining / 60 )) $(( remaining % 60 )))
    fi
    local el
    el=$(printf '%02d:%02d' $(( elapsed / 60 )) $(( elapsed % 60 )))

    printf '\r\033[KNeSVoR |%s| %3d%% (%d/%d) [%s<%s] %s' \
        "$bar" "$frac" "$current" "$total" "$el" "$eta" "$label"
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
DONE=0
SKIPPED=0
FAILED=0

echo "Input : $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Log   : $LOG_FILE"
echo "Found $TOTAL subject/session reconstruction task(s)."
echo

log "==== NeSVoR batch run started ===="
log "Input=$INPUT_DIR Output=$OUTPUT_DIR Tasks=$TOTAL"

for anat in "${TASKS[@]}"; do
    # Derive sub / ses from the path:  .../sub-XXX/ses-YY/anat
    ses_dir=$(dirname "$anat")
    sub_dir=$(dirname "$ses_dir")
    ses=$(basename "$ses_dir")
    sub=$(basename "$sub_dir")

    # Handle datasets without a session level (anat directly under sub-XXX)
    if [[ "$sub" != sub-* ]]; then
        sub="$ses"        # in that case ses_dir was actually the subject dir
        ses=""
    fi

    if [[ -n "$ses" ]]; then
        label="${sub}/${ses}"
        out_anat="${OUTPUT_DIR}/${sub}/${ses}/anat"
        prefix="${sub}_${ses}"
    else
        label="${sub}"
        out_anat="${OUTPUT_DIR}/${sub}/anat"
        prefix="${sub}"
    fi

    draw_bar "$DONE" "$TOTAL" "$label"

    # Gather all T2w stacks for this session
    mapfile -t STACKS < <(find "$anat" -maxdepth 1 -name '*T2w*.nii.gz' | sort)
    if [[ ${#STACKS[@]} -eq 0 ]]; then
        log "SKIP  $label : no T2w stacks"
        (( SKIPPED++ )); (( DONE++ )); continue
    fi

    # Build output volume name: keep BIDS entities, drop run-N, add srr-nesvor.
    # Use the acq entity from the first stack if present.
    acq=""
    first=$(basename "${STACKS[0]}")
    if [[ "$first" =~ (acq-[A-Za-z0-9]+) ]]; then
        acq="_${BASH_REMATCH[1]}"
    fi
    out_volume="${out_anat}/${prefix}${acq}_srr-nesvor_T2w.nii.gz"

    if [[ -f "$out_volume" ]]; then
        log "SKIP  $label : output already exists ($out_volume)"
        (( SKIPPED++ )); (( DONE++ )); continue
    fi

    # Translate host paths to their in-container equivalents.
    #   <INPUT_DIR>/...  -> /incoming/...
    #   <MASKS_DIR>/...  -> /masks/...
    #   <OUTPUT_DIR>/... -> /outgoing/...
    C_STACKS=()
    C_MASKS=()
    missing_mask=0
    for s in "${STACKS[@]}"; do
        C_STACKS+=("${CONTAINER_IN}/${s#"$INPUT_DIR"/}")

        if [[ -n "$MASKS_DIR" ]]; then
            # Mask lives at the same relative path under MASKS_DIR, with the
            # stack's basename mapped to its *_T2w_mask.nii.gz counterpart.
            rel="${s#"$INPUT_DIR"/}"
            rel_mask="$(dirname "$rel")/$(mask_name_for "$(basename "$s")")"
            if [[ ! -f "${MASKS_DIR}/${rel_mask}" ]]; then
                log "MISS  $label : no mask ${MASKS_DIR}/${rel_mask}"
                missing_mask=1
            fi
            C_MASKS+=("${CONTAINER_MASKS}/${rel_mask}")
        fi
    done

    if [[ -n "$MASKS_DIR" && "$missing_mask" -eq 1 ]]; then
        log "SKIP  $label : missing one or more stack masks"
        (( SKIPPED++ )); (( DONE++ )); continue
    fi

    c_out_volume="${CONTAINER_OUT}/${out_volume#"$OUTPUT_DIR"/}"

    # Extra outputs (motion-corrected slices, simulated slices, model) mirror the
    # sub/ses/anat layout under <OUTPUT_DIR>/derivatives/nesvor_extras/.
    extras_anat="${OUTPUT_DIR}/derivatives/nesvor_extras/${out_anat#"$OUTPUT_DIR"/}"
    out_slices_dir="${extras_anat}/${prefix}${acq}_srr-nesvor_slices"
    sim_slices_dir="${extras_anat}/${prefix}${acq}_srr-nesvor_simslices"
    out_model="${extras_anat}/${prefix}${acq}_srr-nesvor_model.pt"

    c_out_slices_dir="${CONTAINER_OUT}/${out_slices_dir#"$OUTPUT_DIR"/}"
    c_sim_slices_dir="${CONTAINER_OUT}/${sim_slices_dir#"$OUTPUT_DIR"/}"
    c_out_model="${CONTAINER_OUT}/${out_model#"$OUTPUT_DIR"/}"

    # Assemble the docker command as an array so it can be printed or executed.
    CMD=(docker run --rm --gpus all --ipc=host
        -v "${INPUT_DIR}:${CONTAINER_IN}:ro")
    [[ -n "$MASKS_DIR" ]] && CMD+=(-v "${MASKS_DIR}:${CONTAINER_MASKS}:ro")
    CMD+=(-v "${OUTPUT_DIR}:${CONTAINER_OUT}:rw"
        "$DOCKER_IMAGE"
        nesvor reconstruct
        --input-stacks "${C_STACKS[@]}")
    [[ -n "$MASKS_DIR" ]] && CMD+=(--stack-masks "${C_MASKS[@]}")
    CMD+=(--output-volume "$c_out_volume"
        --output-slices "$c_out_slices_dir"
        --simulated-slices "$c_sim_slices_dir"
        --output-model "$c_out_model"
        --output-resolution "$OUTPUT_RESOLUTION"
        --bias-field-correction
        --registration svort-only
        --batch-size "$BATCH_SIZE"
        --n-levels-bias "$N_LEVELS_BIAS")

    [[ "$DRY_RUN" -eq 0 ]] && mkdir -p "$out_anat" "$out_slices_dir" "$sim_slices_dir"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        # Print the command (clear the progress-bar line first) and move on.
        printf '\r\033[K'
        echo "[DRY-RUN] $label (${#STACKS[@]} stacks):"
        printf '  '
        printf '%q ' "${CMD[@]}"
        echo
        echo
        log "DRY-RUN $label : $(printf '%q ' "${CMD[@]}")"
        (( DONE++ )); continue
    fi

    log "START $label : ${#STACKS[@]} stacks -> $out_volume"
    {
        echo "----------------------------------------------------------------"
        echo "Subject/Session: $label"
        echo "Stacks (${#STACKS[@]}):"
        printf '  %s\n' "${STACKS[@]}"
        echo "Output: $out_volume"
        echo "Extras: $extras_anat (slices / simslices / model.pt)"
        echo "----------------------------------------------------------------"
    } >> "$LOG_FILE"

    if "${CMD[@]}" >> "$LOG_FILE" 2>&1; then
        log "OK    $label"
    else
        rc=$?
        log "FAIL  $label (exit $rc)"
        (( FAILED++ ))
    fi

    (( DONE++ ))
done

draw_bar "$DONE" "$TOTAL" "complete"
echo
echo
echo "Finished: $DONE/$TOTAL processed | skipped: $SKIPPED | failed: $FAILED"
echo "See log: $LOG_FILE"

log "==== NeSVoR batch run finished: done=$DONE skipped=$SKIPPED failed=$FAILED ===="

# Non-zero exit if anything failed, so callers can detect problems.
[[ "$FAILED" -eq 0 ]]
