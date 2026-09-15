"""Tests for the four human-readable formatters in efficiency measures."""

from utils.efficiency.measures.formatting import (
    format_duration,
    format_flops,
    format_mib,
    format_throughput,
)


class TestFormatFlops:
    def test_giga(self) -> None:
        assert format_flops(1_500_000_000) == "1.50 G"

    def test_mega(self) -> None:
        assert format_flops(2_500_000) == "2.50 M"

    def test_kilo(self) -> None:
        assert format_flops(4_500) == "4.50 K"

    def test_small_value_untouched(self) -> None:
        assert format_flops(999) == "999"

    def test_zero(self) -> None:
        assert format_flops(0) == "0"


class TestFormatDuration:
    def test_seconds_only(self) -> None:
        assert format_duration(59) == "59s"

    def test_zero(self) -> None:
        assert format_duration(0) == "0s"

    def test_minutes_and_seconds(self) -> None:
        assert format_duration(61) == "1m 1s"

    def test_hours_minutes_seconds(self) -> None:
        assert format_duration(3661) == "1h 1m 1s"


class TestFormatMib:
    def test_none_renders_dash(self) -> None:
        assert format_mib(None) == "—"

    def test_known_value_with_thousands_separator(self) -> None:
        assert format_mib(1024.4) == "1,024 MiB"


class TestFormatThroughput:
    def test_known_value(self) -> None:
        assert format_throughput(1234.6) == "1,235 interactions/s"

    def test_zero(self) -> None:
        assert format_throughput(0) == "0 interactions/s"
