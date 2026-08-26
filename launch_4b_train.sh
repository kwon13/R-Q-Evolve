#!/bin/bash
# ============================================================================
# R-Q-Evolve 4B Training — One-Shot Launcher
# ============================================================================
#
# 사용법:
#   bash launch_4b_train.sh                    # 기본값으로 실행
#   bash launch_4b_train.sh --config configs/rq_evolve_4b_4gpu.yaml  # config 지정
#   bash launch_4b_train.sh --wandb-offline    # wandb offline 모드
#   bash launch_4b_train.sh --dry-run          # 환경 점검만 하고 학습 시작 안 함
#   bash launch_4b_train.sh --no-auto-merge    # checkpoint 자동 merge 비활성화
#
# 이 스크립트가 하는 일:
#   1. 이전 Ray 프로세스 정리
#   2. GPU 가용성 확인
#   3. 모델 파일 존재 확인 (없으면 HuggingFace에서 다운로드)
#   4. verl 패치 적용 여부 확인
#   5. wandb 로그인 상태 확인
#   6. .env에서 환경변수 로드
#   7. preflight check 실행
#   8. checkpoint auto-merge daemon + 학습 시작
# ============================================================================
set -euo pipefail

# -- 색상 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[  OK]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
separator() { echo -e "${BLUE}$(printf '─%.0s' {1..60})${NC}"; }

# -- 프로젝트 루트 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
cd "$ROOT"

# -- 기본값 ----
CONFIG="configs/rq_evolve_4b_base.yaml"
MODEL_HF_ID="Qwen/Qwen3-4B-Base"
WANDB_ONLINE=true
DRY_RUN=false
AUTO_MERGE=true
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# -- 인자 파싱 ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)       CONFIG="$2"; shift 2 ;;
    --wandb-offline) WANDB_ONLINE=false; shift ;;
    --dry-run)      DRY_RUN=true; shift ;;
    --no-auto-merge) AUTO_MERGE=false; shift ;;
    --gpus)         GPUS="$2"; shift 2 ;;
    --help|-h)
      head -17 "$0" | tail -15
      exit 0
      ;;
    *) log_fail "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

# Resolve the effective (possibly inherited) config before touching Ray or the
# GPUs.  The old launcher hard-coded /root/models/qwen3-4b-base and could try
# to download there even while the configured model already existed elsewhere.
if ! EFFECTIVE_PATHS="$(python3 -c '
import sys
sys.path.insert(0, "src")
from omegaconf import OmegaConf
from rq_evolve.config import load_raw_config

config = load_raw_config(sys.argv[1])
model_path = OmegaConf.select(config, "verl_config.actor_rollout_ref.model.path")
checkpoint_dir = OmegaConf.select(config, "verl_config.trainer.default_local_dir")
if not model_path:
    raise SystemExit("missing verl_config.actor_rollout_ref.model.path")
if not checkpoint_dir:
    raise SystemExit("missing verl_config.trainer.default_local_dir")
print(model_path)
print(checkpoint_dir)
' "$CONFIG")"; then
  log_fail "Config에서 model.path/default_local_dir를 해석하지 못했습니다: $CONFIG"
  exit 1
fi
mapfile -t EFFECTIVE_PATH_ARRAY <<< "$EFFECTIVE_PATHS"
MODEL_LOCAL="${EFFECTIVE_PATH_ARRAY[0]}"
CKPT_DIR="${EFFECTIVE_PATH_ARRAY[1]}"

# ============================================================================
echo ""
separator
echo -e "${BLUE}  R-Q-Evolve 4B Training Launcher${NC}"
separator
echo ""
log_info "Config : $CONFIG"
log_info "Model  : $MODEL_LOCAL"
log_info "Output : $CKPT_DIR"
log_info "GPUs   : $GPUS"
log_info "WandB  : $(if $WANDB_ONLINE; then echo online; else echo offline; fi)"
log_info "Merge  : $(if $AUTO_MERGE; then echo automatic; else echo disabled; fi)"
echo ""

# ============================================================================
# 1. 이전 Ray / GPU 프로세스 정리
# ============================================================================
separator
log_info "Step 1/7: 이전 프로세스 정리"

if ray status &>/dev/null; then
  log_warn "기존 Ray 클러스터 발견 → 정리 중..."
  ray stop --force 2>/dev/null || true
  sleep 2
  log_ok "Ray 정리 완료"
else
  log_ok "기존 Ray 프로세스 없음"
fi

# ============================================================================
# 2. GPU 가용성 확인
# ============================================================================
separator
log_info "Step 2/7: GPU 가용성 확인"

if ! command -v nvidia-smi &>/dev/null; then
  log_fail "nvidia-smi를 찾을 수 없습니다."
  exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader)

echo "$GPU_INFO" | while IFS= read -r line; do
  log_ok "GPU $line"
done

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
REQUIRED=${#GPU_ARRAY[@]}
if [ "$GPU_COUNT" -lt "$REQUIRED" ]; then
  log_fail "GPU ${REQUIRED}개 필요, ${GPU_COUNT}개 사용 가능"
  exit 1
fi
log_ok "GPU ${REQUIRED}개 사용 가능 ✓"

# ============================================================================
# 3. 모델 파일 확인 (없으면 다운로드)
# ============================================================================
separator
log_info "Step 3/7: 모델 파일 확인"

if [ -d "$MODEL_LOCAL" ] && [ -f "$MODEL_LOCAL/config.json" ]; then
  MODEL_SIZE=$(du -sh "$MODEL_LOCAL" 2>/dev/null | cut -f1)
  log_ok "모델 존재: $MODEL_LOCAL ($MODEL_SIZE)"
else
  log_warn "모델이 없습니다. HuggingFace에서 다운로드합니다..."
  log_info "  $MODEL_HF_ID → $MODEL_LOCAL"

  # .env에서 HF_TOKEN 로드
  if [ -f "$ROOT/.env" ]; then
    export HF_TOKEN=$(grep '^HF_TOKEN=' "$ROOT/.env" | cut -d'=' -f2)
  fi

  huggingface-cli download "$MODEL_HF_ID" --local-dir "$MODEL_LOCAL"

  if [ ! -f "$MODEL_LOCAL/config.json" ]; then
    log_fail "모델 다운로드 실패"
    exit 1
  fi
  log_ok "모델 다운로드 완료"
fi

# ============================================================================
# 4. Python 패키지 확인
# ============================================================================
separator
log_info "Step 4/7: Python 패키지 확인"

MISSING_PKGS=()
for pkg in verl ray torch vllm transformers wandb omegaconf; do
  if python3 -c "import $pkg" 2>/dev/null; then
    VER=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>/dev/null)
    log_ok "$pkg ($VER)"
  else
    log_fail "$pkg 미설치"
    MISSING_PKGS+=("$pkg")
  fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
  log_fail "누락된 패키지: ${MISSING_PKGS[*]}"
  log_info "  pip install ${MISSING_PKGS[*]} 로 설치하세요."
  exit 1
fi

# CUDA 확인
CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_OK" != "True" ]; then
  log_fail "PyTorch CUDA 사용 불가"
  exit 1
fi
log_ok "PyTorch CUDA ✓"

# ============================================================================
# 5. verl 패치 확인
# ============================================================================
separator
log_info "Step 5/7: verl 패치 확인"

PATCH_STATUS=$(python3 -c "
import sys; sys.path.insert(0, 'patches')
import verl_agent_loop_sampling as p
print('applied' if p.is_applied() else 'missing')
" 2>/dev/null || echo "error")

if [ "$PATCH_STATUS" = "applied" ]; then
  log_ok "verl agent loop sampling 패치 적용됨 ✓"
elif [ "$PATCH_STATUS" = "missing" ]; then
  log_warn "패치 미적용 → 지금 적용합니다..."
  python3 patches/verl_agent_loop_sampling.py
  log_ok "패치 적용 완료"
else
  log_fail "패치 확인 실패"
  exit 1
fi

# ============================================================================
# 6. WandB 확인
# ============================================================================
separator
log_info "Step 6/7: WandB 로그인 확인"

# .env에서 API key 로드
if [ -f "$ROOT/.env" ]; then
  WANDB_KEY=$(grep '^WANDB_API_KEY=' "$ROOT/.env" | cut -d'=' -f2)
  if [ -n "$WANDB_KEY" ]; then
    export WANDB_API_KEY="$WANDB_KEY"
  fi
fi

if $WANDB_ONLINE; then
  export WANDB_MODE=online

  # .netrc 확인 또는 API key로 로그인
  if [ -f /root/.netrc ] && grep -q "api.wandb.ai" /root/.netrc 2>/dev/null; then
    WANDB_USER=$(python3 -c "import wandb; wandb.login(relogin=False); api = wandb.Api(); print(api.viewer.username)" 2>/dev/null || echo "unknown")
    log_ok "WandB 로그인됨 (user: $WANDB_USER)"
  elif [ -n "${WANDB_API_KEY:-}" ]; then
    log_warn "WandB .netrc 없음 → API key로 로그인"
    python3 -c "import wandb; wandb.login(key='$WANDB_API_KEY')" 2>/dev/null
    log_ok "WandB 로그인 완료"
  else
    log_fail "WandB API key가 없습니다. .env 파일에 WANDB_API_KEY를 설정하세요."
    exit 1
  fi
else
  export WANDB_MODE=offline
  log_ok "WandB offline 모드"
fi

# ============================================================================
# 7. Preflight Check
# ============================================================================
separator
log_info "Step 7/7: Preflight Check"

if ! python3 scripts/preflight_check.py --config "$CONFIG" 2>&1; then
  log_fail "Preflight check 실패! 위의 오류를 확인하세요."
  exit 1
fi
log_ok "Preflight check 통과 ✓"

# ============================================================================
# 학습 시작
# ============================================================================
echo ""
separator
if $DRY_RUN; then
  echo -e "${GREEN}  ✓ Dry-run 완료: 모든 점검 통과.${NC}"
  if $AUTO_MERGE; then
    log_info "실제 실행에서는 $CKPT_DIR 의 이전 checkpoint가 자동 merge됩니다."
  fi
  separator
  exit 0
fi

# 환경변수 설정
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# .env에서 OPENAI_API_KEY 로드
if [ -f "$ROOT/.env" ]; then
  OAI_KEY=$(grep '^OPENAI_API_KEY=' "$ROOT/.env" | cut -d'=' -f2)
  if [ -n "$OAI_KEY" ]; then
    export OPENAI_API_KEY="$OAI_KEY"
  fi
  HF_KEY=$(grep '^HF_TOKEN=' "$ROOT/.env" | cut -d'=' -f2)
  if [ -n "$HF_KEY" ]; then
    export HF_TOKEN="$HF_KEY"
  fi
fi

# 로그 파일 생성
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/rq_evolve_4b_$(date +%Y%m%d_%H%M%S).log"

# 새 checkpoint가 생긴 뒤에만 직전 checkpoint를 merge한다. 따라서 최신
# actor/는 항상 resume용으로 온전히 남는다. merge 결과를 검증한 뒤에만
# 이전 actor/를 삭제하며, verl 자체 보존 수는 config에서 null이어야 한다.
if $AUTO_MERGE; then
  mkdir -p "$CKPT_DIR"
  RUN_KEY="$(basename "$CKPT_DIR")"
  AUTO_MERGE_LOG="$ROOT/logs/${RUN_KEY}_auto_merge.log"
  AUTO_MERGE_PID_FILE="$ROOT/logs/${RUN_KEY}_auto_merge.pid"

  if pgrep -f "auto_merge_checkpoints.py.*${RUN_KEY}" >/dev/null; then
    log_ok "Checkpoint auto-merge daemon이 이미 실행 중입니다: $CKPT_DIR"
  else
    nohup python3 scripts/auto_merge_checkpoints.py \
      --ckpt_dir "$CKPT_DIR" --interval 60 \
      >> "$AUTO_MERGE_LOG" 2>&1 &
    AUTO_MERGE_PID=$!
    echo "$AUTO_MERGE_PID" > "$AUTO_MERGE_PID_FILE"
    sleep 3
    if kill -0 "$AUTO_MERGE_PID" 2>/dev/null; then
      log_ok "Checkpoint auto-merge 시작 (PID $AUTO_MERGE_PID)"
      log_info "Merge log: $AUTO_MERGE_LOG"
    else
      log_fail "Checkpoint auto-merge가 즉시 종료되었습니다: $AUTO_MERGE_LOG"
      exit 1
    fi
  fi
else
  log_warn "Checkpoint auto-merge가 비활성화되었습니다. 디스크 사용량을 직접 관리하세요."
fi

echo -e "${GREEN}  🚀 R-Q-Evolve 4B 학습을 시작합니다!${NC}"
echo ""
log_info "Config  : $CONFIG"
log_info "Output  : $CKPT_DIR"
log_info "GPUs    : $GPUS"
log_info "WandB   : $WANDB_MODE"
log_info "Log     : $LOG"
separator
echo ""

set -o pipefail
python3 scripts/train_with_verl.py --config "$CONFIG" 2>&1 | tee "$LOG"
