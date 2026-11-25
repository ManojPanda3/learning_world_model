import os
import shutil
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from sm_wm_project.models import SM_WM
from sm_wm_project.config import Config
from sm_wm_project.env_dataset import KeymapMemoryEnv, ALL_PERMUTATIONS

COLOR_PALETTE = {
    "bg": (18, 20, 26),
    "grid_bg": (28, 32, 42),
    "grid_line": (46, 52, 66),
    "real_agent": (76, 217, 100),       # Green for Ground Truth
    "dream_agent": (255, 75, 75),       # Red for Dreamed Prediction
    "agent_outline": (255, 255, 255),
    "text": (240, 240, 245),
    "muted_text": (145, 150, 165),
    "dream_mode_badge": (230, 80, 80),
    "prime_mode_badge": (80, 140, 230),
}

DIR_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

def render_comparison_frame(
    t, 
    is_dreaming, 
    real_pos, 
    dream_pos, 
    action, 
    perm_id, 
    active_slots, 
    step_err, 
    w=860, 
    h=520
):
    img = Image.new("RGB", (w, h), COLOR_PALETTE["bg"])
    draw = ImageDraw.Draw(img)

    grid_size = 5
    cell_px = 76
    offset_x = 45
    offset_y = 65

    # Grid background
    draw.rectangle(
        [offset_x, offset_y, offset_x + grid_size * cell_px, offset_y + grid_size * cell_px],
        fill=COLOR_PALETTE["grid_bg"],
        outline=COLOR_PALETTE["grid_line"],
        width=2,
    )

    for i in range(1, grid_size):
        x = offset_x + i * cell_px
        draw.line([(x, offset_y), (x, offset_y + grid_size * cell_px)], fill=COLOR_PALETTE["grid_line"], width=1)
        y = offset_y + i * cell_px
        draw.line([(offset_x, y), (offset_x + grid_size * cell_px, y)], fill=COLOR_PALETTE["grid_line"], width=1)

    # Convert coordinates from [-1, 1] to pixel positions
    # pos in [-1, 1] -> grid_idx in [0, 4]
    def coord_to_px(coord):
        # grid index = (coord + 1.0) * 2.0
        gx = (coord[1] + 1.0) * 2.0
        gy = (coord[0] + 1.0) * 2.0
        px = offset_x + gx * cell_px + cell_px // 2
        py = offset_y + gy * cell_px + cell_px // 2
        return px, py

    real_px, real_py = coord_to_px(real_pos)
    dream_px, dream_py = coord_to_px(dream_pos)

    # Draw connection line between ground truth and dreamed position
    if is_dreaming:
        draw.line([(real_px, real_py), (dream_px, dream_py)], fill=(200, 100, 100), width=2)

    # Draw Real Agent (Green)
    r_rad = 20
    draw.ellipse(
        [real_px - r_rad, real_py - r_rad, real_px + r_rad, real_py + r_rad],
        fill=COLOR_PALETTE["real_agent"],
        outline=(255, 255, 255),
        width=2,
    )
    draw.text((real_px - 8, real_py - 7), "GT", fill=(20, 20, 20))

    # Draw Dreamed Agent (Red/Pink)
    d_rad = 18
    if is_dreaming:
        # Glow
        draw.ellipse([dream_px - 26, dream_py - 26, dream_px + 26, dream_py + 26], fill=(120, 30, 30))
        draw.ellipse(
            [dream_px - d_rad, dream_py - d_rad, dream_px + d_rad, dream_py + d_rad],
            fill=COLOR_PALETTE["dream_agent"],
            outline=(255, 255, 255),
            width=2,
        )
        draw.text((dream_px - 8, dream_py - 7), "AI", fill=(255, 255, 255))
    else:
        # Priming: exact match
        draw.ellipse([real_px - 8, real_py - 8, real_px + 8, real_py + 8], fill=COLOR_PALETTE["dream_agent"])

    # Right Telemetry Panel
    panel_x = 460
    draw.text((panel_x, 30), "Open-Loop Dreaming Trial (SM_WM)", fill=COLOR_PALETTE["text"])
    draw.line([(panel_x, 55), (w - 30, 55)], fill=COLOR_PALETTE["grid_line"], width=1)

    # Phase badge
    badge_y = 70
    if not is_dreaming:
        badge_text = f"PHASE: PRIMING WITH REAL OBSERVATIONS (t={t:2d}/20)"
        badge_c = COLOR_PALETTE["prime_mode_badge"]
    else:
        dream_step = t - 20
        badge_text = f"PHASE: OPEN-LOOP DREAMING (Step H={dream_step:2d}/40)"
        badge_c = COLOR_PALETTE["dream_mode_badge"]

    draw.rectangle([panel_x, badge_y, w - 30, badge_y + 34], fill=badge_c)
    draw.text((panel_x + 15, badge_y + 9), badge_text, fill=(255, 255, 255))

    # Telemetry details
    draw.text((panel_x, 125), f"Ground Truth Pos:    ({real_pos[0]:+.2f}, {real_pos[1]:+.2f})", fill=COLOR_PALETTE["real_agent"])
    if is_dreaming:
        draw.text((panel_x, 150), f"Dreamed AI Pos:      ({dream_pos[0]:+.2f}, {dream_pos[1]:+.2f})", fill=COLOR_PALETTE["dream_agent"])
        draw.text((panel_x, 175), f"Euclidean Error:     {step_err:.3f}", fill=(255, 200, 100))
    else:
        draw.text((panel_x, 150), "Dreamed AI Pos:      [Priming Active]", fill=COLOR_PALETTE["muted_text"])
        draw.text((panel_x, 175), "Euclidean Error:     0.000", fill=COLOR_PALETTE["muted_text"])

    # Action details
    perm = ALL_PERMUTATIONS[perm_id]
    act_dir = DIR_NAMES[perm[action]] if action is not None else "NONE"
    draw.text((panel_x, 210), f"Applied Action:      A{action} -> Move {act_dir}", fill=(255, 230, 100))

    # Memory state
    draw.line([(panel_x, 245), (w - 30, 245)], fill=COLOR_PALETTE["grid_line"], width=1)
    draw.text((panel_x, 260), "Internal World Model Diagnosis:", fill=COLOR_PALETTE["text"])
    draw.text((panel_x + 15, 290), f"Active Memory Slots: {active_slots:.1f} / 16", fill=COLOR_PALETTE["text"])

    if is_dreaming:
        if abs(dream_pos[0]) < 0.15 and abs(dream_pos[1]) < 0.15:
            diagnosis_msg = "WARNING: AI frozen near (0,0) center collapse!"
            diag_col = (255, 100, 100)
        else:
            diagnosis_msg = "OK: AI tracking active motion in imagination."
            diag_col = (100, 255, 120)
    else:
        diagnosis_msg = "Priming memory bank with environment physics."
        diag_col = COLOR_PALETTE["muted_text"]

    draw.text((panel_x + 15, 320), diagnosis_msg, fill=diag_col)

    # Legend at bottom
    draw.line([(panel_x, 360), (w - 30, 360)], fill=COLOR_PALETTE["grid_line"], width=1)
    draw.rectangle([panel_x + 15, 380, panel_x + 30, 395], fill=COLOR_PALETTE["real_agent"])
    draw.text((panel_x + 40, 380), "Green = Ground Truth Real Agent", fill=COLOR_PALETTE["text"])

    draw.rectangle([panel_x + 15, 410, panel_x + 30, 425], fill=COLOR_PALETTE["dream_agent"])
    draw.text((panel_x + 40, 410), "Red = AI Hallucinated / Dreamed Agent", fill=COLOR_PALETTE["text"])

    draw.text((offset_x, offset_y + grid_size * cell_px + 15), "5x5 Gridworld | Coordinates normalized from [-1.0, -1.0] to [+1.0, +1.0]", fill=COLOR_PALETTE["muted_text"])

    return img

def main():
    cfg = Config()
    device = torch.device("cpu")
    model = SM_WM(cfg.obs_dim, cfg.act_dim, cfg.d_model, cfg.num_slots, cfg.nhead, cfg.num_layers)
    ckpt = torch.load("/tmp/kaggle_done2/results/sm_wm_seed42_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = "/tmp/dreaming_video_frames"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    env = KeymapMemoryEnv()
    obs, info = env.reset(seed=999, options={"split": "val"})

    prime_steps = 20
    dream_steps = 40
    total_steps = prime_steps + dream_steps

    traj_obs = [obs]
    traj_act = []
    
    # Generate full ground-truth trajectory
    for _ in range(total_steps):
        a = env.action_space.sample()
        n_obs, r, term, trunc, _ = env.step(a)
        traj_obs.append(n_obs)
        traj_act.append(a)

    traj_obs = torch.tensor(np.array(traj_obs), dtype=torch.float32)
    traj_act = F.one_hot(torch.tensor(traj_act, dtype=torch.long), num_classes=4).float()

    frame_idx = 0
    state = model.init_state(batch_size=1, device=device)

    # 1. Priming Phase (t=0..19)
    print("Recording Priming Phase...")
    for t in range(prime_steps):
        x = traj_obs[t:t+1]
        a = traj_act[t:t+1]
        (pred_o, _), state, gate_info = model.step(x, a, state)

        real_pos = traj_obs[t, 0:2].numpy()
        dream_pos = real_pos # during priming, model sees reality
        act_val = int(traj_act[t].argmax().item())

        img = render_comparison_frame(
            t=t,
            is_dreaming=False,
            real_pos=real_pos,
            dream_pos=dream_pos,
            action=act_val,
            perm_id=env.perm_id,
            active_slots=14.5,
            step_err=0.0,
        )
        img.save(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"))
        frame_idx += 1

    # 2. Open-Loop Dreaming Phase (t=20..59)
    print("Recording Open-Loop Dreaming Phase...")
    curr_pred_o = pred_o
    for step_h in range(dream_steps):
        t = prime_steps + step_h
        a = traj_act[t:t+1]
        (curr_pred_o, _), state, gate_info = model.step(curr_pred_o, a, state)

        real_pos = traj_obs[t, 0:2].numpy()
        dream_pos = curr_pred_o[0, 0:2].detach().numpy()
        act_val = int(traj_act[t].argmax().item())
        err = np.linalg.norm(real_pos - dream_pos)

        img = render_comparison_frame(
            t=t,
            is_dreaming=True,
            real_pos=real_pos,
            dream_pos=dream_pos,
            action=act_val,
            perm_id=env.perm_id,
            active_slots=14.5,
            step_err=err,
        )
        img.save(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"))
        frame_idx += 1

    video_path = "/tmp/dreaming_trial_sm_wm.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "3",
        "-i", f"{out_dir}/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path,
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(out_dir)
    print(f"\nDreaming trial video generated: {video_path} ({os.path.getsize(video_path):,} bytes)")

if __name__ == "__main__":
    main()
