"""Real-order replay scenario for EV charging training.

This module intentionally uses only the Python standard library plus numpy.
The project virtualenv does not include pandas/openpyxl, so the loader reads
the required worksheet directly from the xlsx zip/xml structure.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np


COL_ORDER_TIME = "\u4e0b\u5355\u65f6\u95f4"
COL_START_TIME = "\u5f00\u59cb\u5145\u7535\u65f6\u95f4"
COL_ENERGY_KWH = "\u5145\u7535\u7535\u91cf\uff08kWh\uff09"
COL_DURATION_MIN = "\u5145\u7535\u65f6\u957f\uff08\u5206\u949f\uff09"
COL_POWER_KW = "\u5145\u7535\u6869\u529f\u7387"
COL_STATUS = "\u5145\u7535\u72b6\u6001"
COL_START_SOC = "\u5f00\u59cbSOC\uff08%\uff09"
COL_END_SOC = "\u7ed3\u675fSOC\uff08%\uff09"
COL_ELECTRICITY_FEE = "\u8ba2\u5355\u7535\u8d39\uff08\u5143\uff09"
COL_SERVICE_FEE = "\u8ba2\u5355\u670d\u52a1\u8d39\uff08\u5143\uff09"
STATUS_DONE = "\u5df2\u5b8c\u6210"


@dataclass(frozen=True)
class RealOrderRecord:
    """One historical charging order mapped into an episode-day timeline."""

    time_s: float
    energy_kwh: float
    duration_min: Optional[float] = None
    start_soc: Optional[float] = None
    end_soc: Optional[float] = None
    battery_kwh: Optional[float] = None
    pile_power_kw: Optional[float] = None
    electricity_fee_per_kwh: Optional[float] = None
    service_fee_per_kwh: Optional[float] = None


@dataclass(frozen=True)
class RealOrderDataset:
    """Historical orders grouped by calendar day."""

    records_by_day: Dict[date, List[RealOrderRecord]]
    selected_month: Optional[str]
    total_rows: int
    usable_records: int

    @property
    def days(self) -> List[date]:
        return sorted(self.records_by_day)


_DATASET_CACHE: Dict[Tuple[Any, ...], RealOrderDataset] = {}


class RealOrderScenario:
    """Replay historical order arrivals while keeping the market LMP process."""

    def __init__(
        self,
        data_path: str,
        sheet_name: str = "Sheet2",
        arrival_scale: float = 1.0,
        date_filter: str = "dominant_month",
        min_energy_kwh: float = 0.1,
        lmp_base: float = 0.20,
        lmp_amp: float = 0.10,
        price_freq: float = 3600.0,
        use_order_lmp: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.data_path = str(Path(data_path))
        self.sheet_name = sheet_name
        self.arrival_scale = max(float(arrival_scale), 0.0)
        self.date_filter = date_filter
        self.min_energy_kwh = float(min_energy_kwh)
        self.lmp_base = float(lmp_base)
        self.lmp_amp = float(lmp_amp)
        self.price_freq = float(price_freq)
        self.use_order_lmp = bool(use_order_lmp)
        self.rng = np.random.default_rng(seed)

        self.dataset = load_real_order_dataset(
            data_path=self.data_path,
            sheet_name=self.sheet_name,
            date_filter=self.date_filter,
            min_energy_kwh=self.min_energy_kwh,
        )
        if not self.dataset.days:
            raise ValueError(f"No usable real orders loaded from {self.data_path!r}.")

        self.time_seconds = 0.0
        self.last_price_update = -self.price_freq
        self.current_lmp = self.lmp_base
        self.active_day: date = self.dataset.days[0]
        self._active_records: List[RealOrderRecord] = []
        self._record_index = 0
        self._arrival_carry = 0.0
        self._pending_records: List[RealOrderRecord] = []
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        days = self.dataset.days
        day_index = int(seed or 0) % len(days)
        self.active_day = days[day_index]
        self._active_records = self.dataset.records_by_day[self.active_day]
        self._record_index = 0
        self._arrival_carry = 0.0
        self._pending_records = []
        self.time_seconds = 0.0
        self.last_price_update = -self.price_freq
        self.current_lmp = self.lmp_base

    def step(self, dt: float) -> Dict[str, Any]:
        previous_time = self.time_seconds
        self.time_seconds += float(dt)

        interval_records: List[RealOrderRecord] = []
        while self._record_index < len(self._active_records):
            record = self._active_records[self._record_index]
            if record.time_s > self.time_seconds:
                break
            self._record_index += 1
            if record.time_s > previous_time or previous_time == 0.0:
                interval_records.append(record)

        self._pending_records = self._scale_interval_records(interval_records)

        if self.time_seconds - self.last_price_update >= self.price_freq:
            self.current_lmp = self.lmp_base + self.lmp_amp * np.sin(
                2 * np.pi * self.time_seconds / 86400.0
            )
            self.last_price_update = self.time_seconds

        if self.use_order_lmp and interval_records:
            lmp_values = [
                r.electricity_fee_per_kwh
                for r in interval_records
                if r.electricity_fee_per_kwh is not None
            ]
            if lmp_values:
                self.current_lmp = float(np.clip(np.mean(lmp_values), 0.0, 2.0))

        return {
            "lmp": self.current_lmp,
            "t": self.time_seconds,
            "arrivals": len(self._pending_records),
            "real_order_day": self.active_day.isoformat(),
            "raw_real_order_arrivals": len(interval_records),
        }

    def pop_order_record(self) -> Optional[RealOrderRecord]:
        if not self._pending_records:
            return None
        return self._pending_records.pop(0)

    def _scale_interval_records(self, records: List[RealOrderRecord]) -> List[RealOrderRecord]:
        if not records or self.arrival_scale <= 0.0:
            return []
        if abs(self.arrival_scale - 1.0) < 1e-9:
            return list(records)

        expected = len(records) * self.arrival_scale + self._arrival_carry
        count = int(expected)
        self._arrival_carry = expected - count
        if count <= 0:
            return []
        if count <= len(records):
            indices = self.rng.choice(len(records), size=count, replace=False)
            return [records[int(i)] for i in sorted(indices)]

        extra_count = count - len(records)
        extra_indices = self.rng.integers(0, len(records), size=extra_count)
        scaled = list(records)
        scaled.extend(records[int(i)] for i in extra_indices)
        scaled.sort(key=lambda r: r.time_s)
        return scaled


def load_real_order_dataset(
    data_path: str,
    sheet_name: str = "Sheet2",
    date_filter: str = "dominant_month",
    min_energy_kwh: float = 0.1,
) -> RealOrderDataset:
    cache_key = (str(Path(data_path).resolve()), sheet_name, date_filter, float(min_energy_kwh))
    if cache_key not in _DATASET_CACHE:
        _DATASET_CACHE[cache_key] = _load_real_order_dataset_uncached(
            data_path=data_path,
            sheet_name=sheet_name,
            date_filter=date_filter,
            min_energy_kwh=min_energy_kwh,
        )
    return _DATASET_CACHE[cache_key]


def _load_real_order_dataset_uncached(
    data_path: str,
    sheet_name: str,
    date_filter: str,
    min_energy_kwh: float,
) -> RealOrderDataset:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Real order data file not found: {path}")

    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        worksheet_path = _worksheet_path_for_sheet(zf, sheet_name)
        rows = _iter_xlsx_rows(zf, worksheet_path, shared_strings)
        try:
            header_row = next(rows)
        except StopIteration as exc:
            raise ValueError(f"Worksheet {sheet_name!r} is empty in {path}") from exc

        headers = {str(value): idx for idx, value in enumerate(header_row)}
        records: List[Tuple[datetime, RealOrderRecord]] = []
        total_rows = 0

        for row in rows:
            total_rows += 1
            record_pair = _parse_order_row(
                row=row,
                headers=headers,
                min_energy_kwh=min_energy_kwh,
            )
            if record_pair is not None:
                records.append(record_pair)

    selected_month = _select_month(records, date_filter)
    grouped: Dict[date, List[RealOrderRecord]] = defaultdict(list)
    for started_at, record in records:
        if selected_month and started_at.strftime("%Y-%m") != selected_month:
            continue
        grouped[started_at.date()].append(record)

    records_by_day = {
        day: sorted(day_records, key=lambda r: r.time_s)
        for day, day_records in grouped.items()
        if day_records
    }
    usable_records = sum(len(day_records) for day_records in records_by_day.values())
    return RealOrderDataset(
        records_by_day=records_by_day,
        selected_month=selected_month,
        total_rows=total_rows,
        usable_records=usable_records,
    )


def _parse_order_row(
    row: List[Any],
    headers: Dict[str, int],
    min_energy_kwh: float,
) -> Optional[Tuple[datetime, RealOrderRecord]]:
    status = _row_value(row, headers, COL_STATUS)
    if status not in (None, "", STATUS_DONE):
        return None

    started_at = _parse_datetime(
        _row_value(row, headers, COL_START_TIME)
        or _row_value(row, headers, COL_ORDER_TIME)
    )
    if started_at is None:
        return None

    energy_kwh = _to_float(_row_value(row, headers, COL_ENERGY_KWH))
    if energy_kwh is None or energy_kwh < min_energy_kwh:
        return None

    duration_min = _to_float(_row_value(row, headers, COL_DURATION_MIN))
    start_soc_pct = _to_float(_row_value(row, headers, COL_START_SOC))
    end_soc_pct = _to_float(_row_value(row, headers, COL_END_SOC))
    pile_power_kw = _to_float(_row_value(row, headers, COL_POWER_KW))
    electricity_fee = _to_float(_row_value(row, headers, COL_ELECTRICITY_FEE))
    service_fee = _to_float(_row_value(row, headers, COL_SERVICE_FEE))

    start_soc = _soc_pct_to_frac(start_soc_pct)
    end_soc = _soc_pct_to_frac(end_soc_pct)
    battery_kwh = _estimate_battery_kwh(energy_kwh, start_soc, end_soc)

    time_s = (
        started_at.hour * 3600.0
        + started_at.minute * 60.0
        + started_at.second
        + started_at.microsecond / 1_000_000.0
    )
    record = RealOrderRecord(
        time_s=float(time_s),
        energy_kwh=float(energy_kwh),
        duration_min=duration_min,
        start_soc=start_soc,
        end_soc=end_soc,
        battery_kwh=battery_kwh,
        pile_power_kw=pile_power_kw,
        electricity_fee_per_kwh=_fee_per_kwh(electricity_fee, energy_kwh),
        service_fee_per_kwh=_fee_per_kwh(service_fee, energy_kwh),
    )
    return started_at, record


def _select_month(
    records: Iterable[Tuple[datetime, RealOrderRecord]],
    date_filter: str,
) -> Optional[str]:
    normalized = (date_filter or "dominant_month").strip().lower()
    if normalized in {"", "all", "none"}:
        return None
    if re.fullmatch(r"\d{4}-\d{2}", normalized):
        return normalized
    if normalized != "dominant_month":
        raise ValueError(
            "real_order_date_filter must be 'dominant_month', 'all', or YYYY-MM."
        )

    month_counts = Counter(started_at.strftime("%Y-%m") for started_at, _ in records)
    if not month_counts:
        return None
    return month_counts.most_common(1)[0][0]


def _row_value(row: List[Any], headers: Dict[str, int], column_name: str) -> Any:
    index = headers.get(column_name)
    if index is None or index >= len(row):
        return None
    return row[index]


def _soc_pct_to_frac(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    frac = value / 100.0 if value > 1.0 else value
    if not np.isfinite(frac):
        return None
    return float(np.clip(frac, 0.0, 1.0))


def _estimate_battery_kwh(
    energy_kwh: float,
    start_soc: Optional[float],
    end_soc: Optional[float],
) -> Optional[float]:
    if start_soc is None or end_soc is None:
        return None
    delta_soc = end_soc - start_soc
    if delta_soc <= 0.02:
        return None
    estimated = energy_kwh / delta_soc
    if not np.isfinite(estimated):
        return None
    return float(np.clip(estimated, 35.0, 150.0))


def _fee_per_kwh(fee: Optional[float], energy_kwh: float) -> Optional[float]:
    if fee is None or energy_kwh <= 1e-9:
        return None
    value = fee / energy_kwh
    if not np.isfinite(value):
        return None
    return float(value)


def _read_shared_strings(zf: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    with zf.open("xl/sharedStrings.xml") as fh:
        root = ET.parse(fh).getroot()
    strings = []
    for si in root:
        if _local_name(si.tag) != "si":
            continue
        text_parts = [
            text_node.text or ""
            for text_node in si.iter()
            if _local_name(text_node.tag) == "t"
        ]
        strings.append("".join(text_parts))
    return strings


def _worksheet_path_for_sheet(zf: ZipFile, sheet_name: str) -> str:
    with zf.open("xl/workbook.xml") as fh:
        workbook_root = ET.parse(fh).getroot()
    with zf.open("xl/_rels/workbook.xml.rels") as fh:
        rels_root = ET.parse(fh).getroot()

    rel_targets = {}
    for rel in rels_root:
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            rel_targets[rel_id] = target

    for sheet in workbook_root.iter():
        if _local_name(sheet.tag) != "sheet":
            continue
        if sheet.attrib.get("name") != sheet_name:
            continue
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_targets.get(rel_id)
        if not target:
            break
        if target.startswith("/"):
            return target.lstrip("/")
        if target.startswith("xl/"):
            return target
        return f"xl/{target}"

    available = [
        sheet.attrib.get("name")
        for sheet in workbook_root.iter()
        if _local_name(sheet.tag) == "sheet"
    ]
    raise ValueError(f"Worksheet {sheet_name!r} not found. Available sheets: {available}")


def _iter_xlsx_rows(
    zf: ZipFile,
    worksheet_path: str,
    shared_strings: List[str],
) -> Iterable[List[Any]]:
    with zf.open(worksheet_path) as fh:
        for event, elem in ET.iterparse(fh, events=("end",)):
            if _local_name(elem.tag) != "row":
                continue
            values_by_col: Dict[int, Any] = {}
            max_col = -1
            for cell in elem:
                if _local_name(cell.tag) != "c":
                    continue
                col_idx = _column_index(cell.attrib.get("r", ""))
                if col_idx is None:
                    continue
                max_col = max(max_col, col_idx)
                values_by_col[col_idx] = _parse_cell_value(cell, shared_strings)
            yield [values_by_col.get(i) for i in range(max_col + 1)]
            elem.clear()


def _parse_cell_value(cell: ET.Element, shared_strings: List[str]) -> Any:
    cell_type = cell.attrib.get("t")
    value_text = None
    for child in cell:
        child_name = _local_name(child.tag)
        if child_name == "v":
            value_text = child.text
            break
        if child_name == "is":
            text_parts = [
                text_node.text or ""
                for text_node in child.iter()
                if _local_name(text_node.tag) == "t"
            ]
            return "".join(text_parts)

    if value_text is None:
        return None
    if cell_type == "s":
        index = int(float(value_text))
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
        return ""
    if cell_type in {"str", "inlineStr"}:
        return value_text
    if cell_type == "b":
        return bool(int(float(value_text)))
    return _to_float(value_text)


def _column_index(cell_ref: str) -> Optional[int]:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return None
    col = 0
    for char in match.group(1):
        col = col * 26 + ord(char) - ord("A") + 1
    return col - 1


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    number = _to_float(value)
    if number is not None:
        # Excel's Windows date system is represented with a 1899-12-30 epoch.
        return datetime(1899, 12, 30) + timedelta(days=float(number))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value_float = float(value)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value_float = float(text)
        except ValueError:
            return None
        return value_float if np.isfinite(value_float) else None
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
