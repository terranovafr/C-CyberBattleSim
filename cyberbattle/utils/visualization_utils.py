import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import networkx as nx
from cyberbattle.simulation import model as attacker_model # noqa: E402
import copy

def create_interactive_agent_visualization(model, env, num_episodes=5):
    # === Collect trajectories for multiple episodes ===
    all_episodes = []
    for ep in range(num_episodes):
        trajectory = []
        state = env.reset()
        done = False
        starter_node = env.envs[0].current_env.starter_node
        exploited_records = []  # track (src, tgt, vuln_id, outcome)

        while not done:
            action, _ = model.predict(state)
            next_state, reward, done, info = env.step(action)
            state = next_state

            owned_nodes, discovered_nodes, _, disrupted_nodes, _, _, _, _, network_availability, _, _, _, _, _ = env.envs[0].current_env.get_statistics()
            G = env.envs[0].current_env.evolving_visible_graph

            info = info[0]
            valid = not isinstance(info['real_outcome_class'], attacker_model.InvalidAction)
            src, tgt, vuln_id, outcome = info['source_node'], info['target_node'], info['vulnerability'], info['outcome']
            exploited_records.append((src, tgt, vuln_id, outcome, valid))
            print("Exploited records:", exploited_records)
            trajectory.append({
                "graph": G.copy(),
                "owned_nodes": copy.deepcopy(env.envs[0].current_env.owned_nodes),
                "discovered_nodes": copy.deepcopy(env.envs[0].current_env.discovered_nodes),
                "disrupted_nodes": [ node_id for node_id in env.envs[0].current_env.discovered_nodes if env.envs[0].current_env.get_node(node_id).status == attacker_model.MachineStatus.Stopped],
                "network_availability": env.envs[0].current_env.network_availability,
                "edges": exploited_records.copy(),
                "starter": starter_node
            })
        all_episodes.append(trajectory)

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id="network-graph", style={"width": "65%", "display": "inline-block", "height": "600px"}),
            html.Div(id="stats-panel",
                     style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "paddingLeft": "20px"})
        ], style={"display": "flex", "justifyContent": "space-between"}),

        html.Div([
            html.Button("Previous Action", id="prev-btn", n_clicks=0,
                        style={"fontSize": "18px", "padding": "10px 20px", "margin": "5px",
                               "backgroundColor": "#4CAF50", "color": "white"}),
            html.Button("Next Action", id="next-btn", n_clicks=0,
                        style={"fontSize": "18px", "padding": "10px 20px", "margin": "5px",
                               "backgroundColor": "#4CAF50", "color": "white"}),
            html.Button("Previous Episode", id="prev-ep-btn", n_clicks=0,
                        style={"fontSize": "18px", "padding": "10px 20px", "margin": "5px",
                               "backgroundColor": "#2196F3", "color": "white"}),
            html.Button("Next Episode", id="next-ep-btn", n_clicks=0,
                        style={"fontSize": "18px", "padding": "10px 20px", "margin": "5px",
                               "backgroundColor": "#2196F3", "color": "white"})
        ], style={
            "position": "fixed",
            "bottom": "10px",
            "left": "50%",
            "transform": "translateX(-50%)",
            "textAlign": "center",
            "zIndex": "1000"
        })
    ])
    current_state = {"episode": 0, "step": 0}

    @app.callback(
        [Output("network-graph", "figure"),
         Output("stats-panel", "children")],
        [Input("prev-btn", "n_clicks"),
         Input("next-btn", "n_clicks"),
         Input("prev-ep-btn", "n_clicks"),
         Input("next-ep-btn", "n_clicks")],
        [State("network-graph", "figure")]
    )
    def update_graph(prev_clicks, next_clicks, prev_ep_clicks, next_ep_clicks, _):
        ctx = dash.callback_context
        if not ctx.triggered:
            pass
        else:
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger == "next-btn" and current_state["step"] < len(all_episodes[current_state["episode"]]) - 1:
                current_state["step"] += 1
            elif trigger == "prev-btn" and current_state["step"] > 0:
                current_state["step"] -= 1
            elif trigger == "next-ep-btn" and current_state["episode"] < len(all_episodes) - 1:
                current_state["episode"] += 1
                current_state["step"] = 0
            elif trigger == "prev-ep-btn" and current_state["episode"] > 0:
                current_state["episode"] -= 1
                current_state["step"] = 0

        step = all_episodes[current_state["episode"]][current_state["step"]]

        # Ensure episode and step are within bounds
        current_state["episode"] = max(0, min(current_state["episode"], len(all_episodes) - 1))
        current_state["step"] = max(0, min(current_state["step"], len(all_episodes[current_state["episode"]]) - 1))

        G = step["graph"]
        pos = nx.circular_layout(G)

        # Node coloring
        node_colors = []
        for node in G.nodes:
            if node == step["starter"]:
                node_colors.append("blue")
            elif node in step["owned_nodes"]:
                node_colors.append("green")
            elif node in step["disrupted_nodes"]:
                node_colors.append("red")
            elif node in step["discovered_nodes"]:
                node_colors.append("lightgray")

        edge_x, edge_y = [], []
        # Exploited edges with CVE labels
        label_traces = []
        vulns_list = {}
        for (src, tgt, vuln_id, outcome, valid) in step["edges"]:
            if src not in G.nodes or tgt not in G.nodes:
                continue
            if src == tgt:
                continue
            if not valid:
                continue
            if (src, tgt) not in vulns_list:
                vulns_list[(src, tgt)] = []
            vulns_list[(src, tgt)].append(vuln_id)

        for (src, tgt), vuln_ids in vulns_list.items():
            if src == tgt:
                continue
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            mx, my = (x0+x1)/2, (y0+y1)/2
            label_traces.append(go.Scatter(
                x=[mx], y=[my],
                mode="text",
                text=[", ".join(vuln_ids)],
                textposition="top center",
                showlegend=False
            ))
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="black"), mode="lines", hoverinfo="none", showlegend=False )

        node_x, node_y = [], []
        node_text = []
        for node in G.nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node.split("_")[1])

        annotations = []
        for i, n in enumerate(G.nodes()):
            x, y = node_x[i], node_y[i]
            if n == step["starter"]:
                txt = "<b>S</b>"
                font = dict(size=16, color="white", family="Arial", weight="bold")
            else:
                txt = str(n)
                font = dict(size=12, color="black", family="Arial")
            annotations.append(dict(
                x=x, y=y, xref="x", yref="y",
                text=txt,
                showarrow=False,
                font=font
            ))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            hoverinfo="text",
            marker=dict(color=node_colors, size=25, line=dict(width=2)),
            showlegend=False)


        fig = go.Figure(data=[edge_trace, node_trace] + label_traces)

        legend_traces = [
            go.Scatter(x=[None], y=[None], mode='markers',
                       marker=dict(size=15, color='green'), name='Owned'),
            go.Scatter(x=[None], y=[None], mode='markers',
                       marker=dict(size=15, color='red'), name='Disrupted'),
            go.Scatter(x=[None], y=[None], mode='markers',
                       marker=dict(size=15, color='blue'), name='Starter'),
            go.Scatter(x=[None], y=[None], mode='markers',
                       marker=dict(size=15, color='lightgray'), name='Discovered'),
        ]

        fig.add_traces(legend_traces)

        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                y=-0.2,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=14)
            ),
            hovermode="closest",
            margin=dict(b=80, l=5, r=5, t=40)
        )

        vuln_rows = [
            html.Tr([html.Th("Target Node"), html.Th("CVE ID"), html.Th("Outcome"), html.Th("Valid")], style={"backgroundColor": "#f0f0f0"})
        ]
        for (src, tgt, vuln_id, outcome, valid) in step["edges"]:
            vuln_rows.append(html.Tr([html.Td(str(tgt)), html.Td(str(vuln_id)), html.Td(str(outcome)), html.Td(str(valid))]))

        stats_panel = html.Div([
            html.H3(f"Episode {current_state['episode'] + 1} / {len(all_episodes)}, Step {current_state['step'] + 1} / {len(all_episodes[current_state['episode']])}", style={"textAlign": "center"}),

            # Metrics table
            html.Div(
                html.Table([
                    html.Tr([html.Th("Metric"), html.Th("Value")], style={"backgroundColor": "#f0f0f0"}),
                    html.Tr([html.Td("Owned Nodes"), html.Td(len(step["owned_nodes"]))]),
                    html.Tr([html.Td("Discovered Nodes"), html.Td(len(step["discovered_nodes"]))]),
                    html.Tr([html.Td("Disrupted Nodes"), html.Td(len(step["disrupted_nodes"]))]),
                    html.Tr([html.Td("Network Availability"), html.Td(f"{step['network_availability']:.2f}")]),
                ], style={"border": "1px solid black", "fontSize": "18px", "width": "100%"}),
                style={"maxHeight": "150px", "overflowY": "auto", "marginBottom": "20px"}
            ),

            # Exploited vulnerabilities table
            html.Div(
                html.Table(vuln_rows, style={"border": "1px solid black", "fontSize": "16px", "width": "100%"}),
                style={"maxHeight": "500px", "overflowY": "auto"}
            )
        ])

        return fig, stats_panel

    app.run(debug=True)


