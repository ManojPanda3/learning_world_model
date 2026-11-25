import os
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sm_wm_project.env_dataset import KeymapMemoryEnv, ALL_PERMUTATIONS

COLOR_PALETTE = {
    "bg": (20, 22, 28),
    "grid_bg": (30, 34, 44),
    "grid_line": (48, 54, 68),
    "agent": (64, 196, 255),
    "agent_outline": (255, 255, 255),
    "text": (240, 240, 245),
    "muted_text": (150, 155, 170),
    "key_colors": [
        (255, 75, 75),    # 0: Red
        (76, 217, 100),   # 1: Green
        (255, 204, 0),    # 2: Yellow
        (175, 82, 222),   # 3: Purple
    ],
    "door_colors": [
        (255, 120, 120),
        (120, 230, 140),
        (255, 220, 80),
        (200, 120, 240),
    ],
    "banner_pickup": (255, 204, 0),
    "banner_door": (76, 217, 100),
    "banner_probe": (64, 196, 255),
    "banner_noise": (80, 86, 100),
}

DIR_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

def draw_key_icon(draw, cx, cy, color, is_picked):
    alpha_color = color if not is_picked else (color[0]//2, color[1]//2, color[2]//2)
    # Outer glow
    draw.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], outline=alpha_color, width=2)
    # Head of key
    draw.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=alpha_color, outline=(255, 255, 255), width=2)
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(30, 34, 44))
    # Shaft & teeth
    draw.rectangle([cx + 6, cy - 3, cx + 22, cy + 3], fill=alpha_color)
    draw.rectangle([cx + 14, cy + 3, cx + 18, cy + 9], fill=alpha_color)
    draw.rectangle([cx + 19, cy + 3, cx + 22, cy + 7], fill=alpha_color)

def draw_door_icon(draw, cx, cy, color, is_unlocked):
    # Outer frame
    frame_color = (255, 255, 255) if is_unlocked else color
    fill_color = (30, 34, 44) if is_unlocked else (color[0]//3, color[1]//3, color[2]//3)
    draw.rectangle([cx - 20, cy - 28, cx + 20, cy + 28], fill=fill_color, outline=frame_color, width=3)
    # Arch top
    draw.arc([cx - 20, cy - 36, cx + 20, cy - 10], start=180, end=360, fill=frame_color, width=3)
    if not is_unlocked:
        # Keyhole
        draw.ellipse([cx - 4, cy - 6, cx + 4, cy + 2], fill=color)
        draw.polygon([(cx - 3, cy), (cx + 3, cy), (cx + 5, cy + 10), (cx - 5, cy + 10)], fill=color)
    else:
        # Open indicator (glow lines)
        draw.line([(cx - 12, cy - 10), (cx + 12, cy + 10)], fill=(76, 217, 100), width=3)
        draw.line([(cx - 12, cy + 10), (cx + 12, cy - 10)], fill=(76, 217, 100), width=3)

def render_frame(env, step_info, action, reward, key_pos, door_pos, w=840, h=520):
    img = Image.new("RGB", (w, h), COLOR_PALETTE["bg"])
    draw = ImageDraw.Draw(img)
    
    # Left: Grid View
    grid_size = 5
    cell_px = 76
    offset_x = 40
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

    key_c = COLOR_PALETTE["key_colors"][env.key_color]
    door_c = COLOR_PALETTE["door_colors"][env.key_color]

    # Always draw KEY at key_pos (until picked up)
    kr, kc = key_pos
    k_cx = offset_x + kc * cell_px + cell_px // 2
    k_cy = offset_y + kr * cell_px + cell_px // 2
    is_picked = (env.t >= env.key_step)
    draw_key_icon(draw, k_cx, k_cy, key_c, is_picked=is_picked)

    # Always draw DOOR at door_pos
    dr, dc = door_pos
    d_cx = offset_x + dc * cell_px + cell_px // 2
    d_cy = offset_y + dr * cell_px + cell_px // 2
    draw_door_icon(draw, d_cx, d_cy, door_c, is_unlocked=env.door_opened)

    # Draw Agent
    r, c = env.pos
    cx = offset_x + c * cell_px + cell_px // 2
    cy = offset_y + r * cell_px + cell_px // 2
    rad = 22
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=COLOR_PALETTE["agent"], outline=COLOR_PALETTE["agent_outline"], width=3)
    
    # If agent is carrying the key, draw the key color center inside agent
    if env.has_key:
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=key_c, outline=(255, 255, 255), width=2)

    # Right: Telemetry Panel
    panel_x = 450
    draw.text((panel_x, 30), "KeymapMemoryEnv (Gymnasium Simulation)", fill=COLOR_PALETTE["text"])
    draw.line([(panel_x, 55), (w - 30, 55)], fill=COLOR_PALETTE["grid_line"], width=1)

    # Coordinates & Step
    draw.text((panel_x, 70), f"Timestep:   {env.t:2d} / {env.max_t}", fill=COLOR_PALETTE["text"])
    draw.text((panel_x, 95), f"Agent Pos:  ({env.pos[0]}, {env.pos[1]})", fill=COLOR_PALETTE["text"])
    
    perm = ALL_PERMUTATIONS[env.perm_id]
    if action is not None:
        act_dir = DIR_NAMES[perm[action]]
        draw.text((panel_x, 125), f"Action:     A{action} -> Move {act_dir}", fill=(255, 230, 100))
    else:
        draw.text((panel_x, 125), f"Action:     Initial Reset", fill=COLOR_PALETTE["muted_text"])
        
    draw.text((panel_x, 150), f"Perm ID:    {env.perm_id} ({'OOD [18..23]' if env.perm_id >= 18 else 'Train/Val [0..17]'})", fill=COLOR_PALETTE["muted_text"])
    draw.text((panel_x, 175), f"Reward:     {reward:+.1f}", fill=(100, 255, 120) if reward > 0 else COLOR_PALETTE["text"])

    # Event badge
    badge_y = 215
    if step_info.get("is_key_pickup", False):
        badge_text = f"EVENT: KEY PICKUP! (Color #{env.key_color})"
        badge_c = COLOR_PALETTE["banner_pickup"]
    elif step_info.get("is_door_open", False):
        badge_text = f"EVENT: DOOR OPENED! (+1.0 REWARD)"
        badge_c = COLOR_PALETTE["banner_door"]
    elif step_info.get("is_novel_probe", False):
        badge_text = f"EVENT: NOVEL ACTION PROBE (A{action})"
        badge_c = COLOR_PALETTE["banner_probe"]
    else:
        badge_text = f"EVENT: Distractor step (t={env.t})"
        badge_c = COLOR_PALETTE["banner_noise"]

    draw.rectangle([panel_x, badge_y, w - 30, badge_y + 36], fill=badge_c)
    draw.text((panel_x + 15, badge_y + 10), badge_text, fill=(20, 22, 28))

    # Sparse Memory Subsystem
    draw.line([(panel_x, 275), (w - 30, 275)], fill=COLOR_PALETTE["grid_line"], width=1)
    draw.text((panel_x, 290), "Memory Slots & Lifecycle:", fill=COLOR_PALETTE["text"])
    
    if env.has_key:
        k_txt = f"STORED IN SLOT (Color #{env.key_color})"
        k_col = key_c
    elif env.door_opened:
        k_txt = f"EVICTED (Consumed at door t={env.door_step})"
        k_col = (100, 255, 120)
    else:
        k_txt = f"EMPTY (Key at {key_pos}, step {env.key_step})"
        k_col = COLOR_PALETTE["muted_text"]
        
    draw.text((panel_x + 15, 320), f"Key Slot:    {k_txt}", fill=k_col)
    
    d_txt = f"OPENED (+1.0 Rew)" if env.door_opened else f"LOCKED (Door at {door_pos}, step {env.door_step})"
    draw.text((panel_x + 15, 345), f"Door State:  {d_txt}", fill=(100, 255, 120) if env.door_opened else COLOR_PALETTE["muted_text"])
    
    probed_str = ", ".join(f"A{a}->{DIR_NAMES[perm[a]]}" for a in sorted(env.probed_actions))
    draw.text((panel_x + 15, 370), f"Discovered:  [{probed_str}]", fill=COLOR_PALETTE["muted_text"])

    # Legend / Labels on Grid
    draw.text((offset_x, offset_y + grid_size * cell_px + 12), f"Key (t={env.key_step}): ({kr},{kc})", fill=key_c)
    draw.text((offset_x + 160, offset_y + grid_size * cell_px + 12), f"Door (t={env.door_step}): ({dr},{dc})", fill=door_c)

    return img

def main():
    out_dir = "/tmp/env_video_frames"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    env = KeymapMemoryEnv()
    obs, info = env.reset(seed=123, options={"split": "train"})
    
    # In this run: env.key_step = 5, env.door_step = 44
    # Let's find agent's position at step 5 and step 44
    key_pos = (1, 3)
    door_pos = (4, 4)

    # Frame 0
    img = render_frame(env, info, action=None, reward=0.0, key_pos=key_pos, door_pos=door_pos)
    img.save(os.path.join(out_dir, f"frame_{0:04d}.png"))
    
    frame_idx = 1
    rng = np.random.default_rng(42)
    
    for t in range(env.max_t):
        # Steer agent toward key around t=5, and toward door around t=44
        if t == 4:
            # Force agent onto key position at t=5
            env.pos = np.array(key_pos, dtype=np.int32)
        elif t == 43:
            # Force agent onto door position at t=44
            env.pos = np.array(door_pos, dtype=np.int32)
            
        a = int(rng.integers(0, 4))
        obs, rew, terminated, truncated, info = env.step(a)
        
        # When t == 5, update key_pos to actual pos
        if info.get("is_key_pickup", False):
            key_pos = (env.pos[0], env.pos[1])
        if info.get("is_door_open", False):
            door_pos = (env.pos[0], env.pos[1])

        img = render_frame(env, info, action=a, reward=rew, key_pos=key_pos, door_pos=door_pos)
        img.save(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"))
        frame_idx += 1
        
        # Hold key pickup for 4 frames
        if info.get("is_key_pickup", False):
            for _ in range(4):
                img.save(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"))
                frame_idx += 1
                
        # Hold door opened for 6 frames
        if info.get("is_door_open", False):
            for _ in range(6):
                img.save(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"))
                frame_idx += 1
                
        if truncated:
            break
            
    video_path = "/tmp/collect_dataset_demo.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "3",
        "-i", f"{out_dir}/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(out_dir)
    print(f"Video saved to {video_path} ({os.path.getsize(video_path):,} bytes)")

if __name__ == "__main__":
    main()
