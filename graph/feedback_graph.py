"""Performance feedback graph: performance_monitor -> learning. Triggered by the scheduler at
the 24h and 7d marks after a video publishes — deliberately decoupled from the publish graph
in graph/builder.py since it fires on a delay, not inline with the run that published it."""
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from graph.nodes.feedback import learning_node, performance_monitor_node


class FeedbackState(TypedDict, total=False):
    youtube_video_id: str
    window: str  # "24h" | "7d"
    performance_snapshot: dict
    learning_update: dict


def build_feedback_graph() -> StateGraph:
    graph = StateGraph(FeedbackState)
    graph.add_node("performance_monitor", performance_monitor_node)
    graph.add_node("learning", learning_node)
    graph.add_edge(START, "performance_monitor")
    graph.add_edge("performance_monitor", "learning")
    graph.add_edge("learning", END)
    return graph
