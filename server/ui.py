import os
import math
import gradio as gr
import pandas as pd
from .lmd_environment import LmdEnvironment
from .models import LmdAction, OrderStatus

# Import inference helpers
try:
    from inference import _greedy_fallback, _build_prompt, _call_llm
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inference import _greedy_fallback, _build_prompt, _call_llm

class GradioEnv:
    def __init__(self):
        self.env = LmdEnvironment(difficulty="easy")
        self.last_obs = self.env.reset()
        self.history = []

    def reset(self, difficulty):
        self.env = LmdEnvironment(difficulty=difficulty)
        self.last_obs = self.env.reset()
        self.history = [{"Step": 0, "Action": "Reset", "Reward": 0.0, "Message": "Environment Started"}]
        return self.get_ui_updates()

    def step(self, order_id, vehicle_id):
        if self.last_obs.done:
            return self.get_ui_updates()
        
        action = LmdAction(order_id=order_id, vehicle_id=vehicle_id)
        self.last_obs = self.env.step(action)
        self.history.append({
            "Step": self.env._state.step_count,
            "Action": f"Deliver {order_id} via {vehicle_id}",
            "Reward": self.last_obs.reward,
            "Message": self.last_obs.message.split("\n")[0]
        })
        return self.get_ui_updates()

    def agent_step(self):
        if self.last_obs.done:
            return self.get_ui_updates()

        hf_token = os.environ.get("HF_TOKEN")
        decision = None
        if hf_token:
            decision = _call_llm(_build_prompt(self.last_obs, self.env._difficulty))
        
        if not decision or not decision.get("order_id"):
            decision = _greedy_fallback(self.last_obs)
        
        return self.step(decision.get("order_id"), decision.get("vehicle_id"))

    def get_ui_updates(self):
        map_html = f"<pre style='font-family: monospace; line-height: 1.2; background: #1a1a1a; color: #00ff00; padding: 15px; border-radius: 8px;'>{self.env._render_ascii_map()}</pre>"
        
        metrics = {
            "Time": f"{self.env._current_time:.1f}h",
            "Weather": self.last_obs.weather.capitalize(),
            "Traffic": f"{self.last_obs.traffic_level}x",
            "Delivered": f"{self.env._delivered_count}/{len(self.env._orders)}",
            "Violations (Cap/Time)": f"{self.env._capacity_violations}/{self.env._time_violations}"
        }
        
        order_data = []
        for o in self.last_obs.orders:
            order_data.append([o.id, f"({o.location[0]:.1f}, {o.location[1]:.1f})", o.weight, f"{o.time_window[0]:.1f}-{o.time_window[1]:.1f}", o.status.value])
        order_df = pd.DataFrame(order_data, columns=["ID", "Location", "Weight", "Window", "Status"])
        
        vehicle_data = []
        for v in self.last_obs.vehicles:
            status = "Broken" if v.is_broken else f"{v.battery_level:.1f}%"
            vehicle_data.append([v.id, f"({v.location[0]:.1f}, {v.location[1]:.1f})", f"{v.capacity:.1f}/{v.max_capacity:.1f}", status])
        vehicle_df = pd.DataFrame(vehicle_data, columns=["ID", "Location", "Capacity", "Battery/Status"])

        pending_orders = [o.id for o in self.last_obs.orders if o.status == OrderStatus.PENDING]
        active_vehicles = [v.id for v in self.last_obs.vehicles if not v.is_broken]

        return (
            map_html, 
            metrics, 
            order_df, 
            vehicle_df, 
            gr.update(choices=pending_orders, value=pending_orders[0] if pending_orders else None),
            gr.update(choices=active_vehicles, value=active_vehicles[0] if active_vehicles else None),
            pd.DataFrame(self.history[-10:]),
            "Game Over!" if self.last_obs.done else f"Step {self.env._state.step_count} In Progress"
        )

def create_ui():
    ge = GradioEnv()
    
    custom_css = """
    .map-container { background: #121212; border: 2px solid #333; border-radius: 12px; }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"), css=custom_css) as demo:
        gr.Markdown("# 🚚 LMD Control Center")
        
        with gr.Row():
            with gr.Column(scale=2):
                map_display = gr.HTML(ge.get_ui_updates()[0], elem_classes=["map-container"])
                status_text = gr.Label(value="Ready")
            with gr.Column(scale=1):
                metrics_display = gr.JSON(value=ge.get_ui_updates()[1], label="Sim Metrics")
                difficulty = gr.Radio(["easy", "medium", "hard"], label="Difficulty", value="easy")
                btn_reset = gr.Button("🔄 Reset", variant="secondary")

        with gr.Row():
            orders_table = gr.DataFrame(ge.get_ui_updates()[2], label="Orders")
            vehicles_table = gr.DataFrame(ge.get_ui_updates()[3], label="Fleet")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    sel_order = gr.Dropdown(label="Order")
                    sel_vehicle = gr.Dropdown(label="Vehicle")
                btn_step = gr.Button("🚀 Dispatch", variant="primary")
            with gr.Column():
                btn_agent = gr.Button("🧠 AI Autopilot", variant="stop")

        log_table = gr.DataFrame(ge.get_ui_updates()[6], label="Logs")

        outputs = [map_display, metrics_display, orders_table, vehicles_table, sel_order, sel_vehicle, log_table, status_text]
        btn_reset.click(ge.reset, inputs=[difficulty], outputs=outputs)
        btn_step.click(ge.step, inputs=[sel_order, sel_vehicle], outputs=outputs)
        btn_agent.click(ge.agent_step, outputs=outputs)
        
    return demo
