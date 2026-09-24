"""Unit coverage for the YouTube WebSub webhook's pure helpers (XML/feed parsing, signature
check) and strategy_node's numeric weight-adjustment formatting -- no DB or network."""
import hashlib
import hmac
from xml.etree import ElementTree

import pytest

from api.routes import webhooks
from graph.nodes.research import _format_weight_adjustments

_NEW_ENTRY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>yt:video:ABC123</id>
    <yt:videoId>ABC123</yt:videoId>
    <yt:channelId>CHAN1</yt:channelId>
    <title>My New Video</title>
    <published>2026-09-01T12:00:00+00:00</published>
    <updated>2026-09-01T12:05:00+00:00</updated>
  </entry>
</feed>"""

_DELETED_ENTRY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:at="http://purl.org/atompub/tombstones/1.0" xmlns="http://www.w3.org/2005/Atom">
  <at:deleted-entry ref="yt:video:GONE99" when="2026-09-02T09:00:00+00:00">
    <link href="http://www.youtube.com/watch?v=GONE99"/>
  </at:deleted-entry>
</feed>"""


def test_iter_events_parses_new_entry():
    events = list(webhooks._iter_events(ElementTree.fromstring(_NEW_ENTRY)))
    assert len(events) == 1
    kind, video_id, title, published = events[0]
    assert (kind, video_id, title) == ("published", "ABC123", "My New Video")
    assert published is not None and published.year == 2026 and published.hour == 12


def test_iter_events_parses_deleted_entry():
    events = list(webhooks._iter_events(ElementTree.fromstring(_DELETED_ENTRY)))
    assert events == [("deleted", "GONE99", "", None)]


def test_parse_dt_handles_zulu_and_garbage():
    assert webhooks._parse_dt("2026-09-01T12:00:00Z").tzinfo is not None
    assert webhooks._parse_dt("not-a-date") is None
    assert webhooks._parse_dt(None) is None


def test_signature_ok(monkeypatch):
    body = b"<feed/>"
    secret = "s3cr3t"
    monkeypatch.setattr(webhooks, "get_settings", lambda: type("S", (), {"youtube_websub_secret": secret})())
    good = "sha1=" + hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()
    assert webhooks._signature_ok(body, good) is True
    assert webhooks._signature_ok(body, "sha1=deadbeef") is False
    assert webhooks._signature_ok(body, "") is False
    assert webhooks._signature_ok(body, "md5=whatever") is False


@pytest.mark.parametrize(
    "adjustments,expected_substr",
    [
        ({}, "No learned strategy-weight adjustments"),
        ({"evergreen_bias": 0.05}, "evergreen_bias = +0.050"),
        ({"evergreen_bias": -0.2}, "evergreen_bias = -0.200"),
    ],
)
def test_format_weight_adjustments(adjustments, expected_substr):
    assert expected_substr in _format_weight_adjustments(adjustments)
