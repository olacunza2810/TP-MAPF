"""Persistencia en parquet particionado.

Se elige parquet columnar con particiones Hive (``underlying=.../trade_date=...``)
porque el backtest lee subconjuntos por ticker y fecha: el predicate pushdown de
``pyarrow`` evita levantar el dataset completo a memoria en cada iteración.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .schemas import enforce_schema

_LOG = logging.getLogger(__name__)

__all__ = ["ParquetStore"]


class ParquetStore:
    """Escritor y lector del data lake local de opciones."""

    def __init__(self, root: Path, dataset_name: str = "option_quotes") -> None:
        """Inicializa el store.

        Args:
            root: Directorio raíz del data lake.
            dataset_name: Subdirectorio del dataset.
        """
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.path = self.root / dataset_name
        self.path.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        frame: pd.DataFrame,
        partition_cols: Sequence[str] = ("underlying", "trade_date"),
        compression: str = "zstd",
    ) -> int:
        """Escribe un lote respetando el esquema canónico.

        Usa ``existing_data_behavior='overwrite_or_ignore'`` con nombres de
        archivo derivados del instante de grabación, de modo que el recorder
        pueda correr en loop agregando particiones sin reescribir lo ya
        persistido. La deduplicación se resuelve en lectura por la clave
        ``(symbol, timestamp)``, no en escritura, para que el proceso de ingesta
        nunca tenga que leer el dataset entero.

        Args:
            frame: Datos a persistir.
            partition_cols: Columnas de partición.
            compression: Códec. ``zstd`` da mejor ratio que ``snappy`` con
                velocidad de descompresión comparable.

        Returns:
            Cantidad de filas escritas.
        """
        if frame.empty:
            _LOG.info("Nada para escribir.")
            return 0

        normalized = enforce_schema(frame)
        normalized["trade_date"] = pd.to_datetime(
            normalized["trade_date"]
        ).dt.strftime("%Y-%m-%d")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        table = pa.Table.from_pandas(normalized, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(self.path),
            partition_cols=list(partition_cols),
            compression=compression,
            basename_template=f"part-{stamp}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        _LOG.info("Escritas %d filas en %s", len(normalized), self.path)
        return len(normalized)

    def read(
        self,
        underlyings: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        columns: Sequence[str] | None = None,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
        """Lee el dataset aplicando filtros de partición.

        Args:
            underlyings: Tickers a leer. ``None`` lee todos.
            start: Fecha mínima ``YYYY-MM-DD`` inclusive.
            end: Fecha máxima ``YYYY-MM-DD`` inclusive.
            columns: Subconjunto de columnas.
            deduplicate: Si ``True``, conserva la última observación por
                ``(symbol, timestamp)``.

        Returns:
            DataFrame con los datos solicitados.
        """
        dataset = ds.dataset(str(self.path), format="parquet", partitioning="hive")
        expression = None
        if underlyings:
            expression = ds.field("underlying").isin(list(underlyings))
        if start:
            clause = ds.field("trade_date") >= start
            expression = clause if expression is None else expression & clause
        if end:
            clause = ds.field("trade_date") <= end
            expression = clause if expression is None else expression & clause

        frame = dataset.to_table(
            filter=expression, columns=list(columns) if columns else None
        ).to_pandas()

        if deduplicate and {"symbol", "timestamp"}.issubset(frame.columns):
            before = len(frame)
            frame = (
                frame.sort_values("observed_at")
                .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
                .reset_index(drop=True)
            )
            if before != len(frame):
                _LOG.info("Deduplicación: %d -> %d filas.", before, len(frame))
        return frame

    def write_manifest(self, manifest: dict[str, Any], name: str = "manifest") -> Path:
        """Persiste los supuestos de la corrida junto al dataset.

        Sin manifiesto, un parquet de opciones es irreproducible: no se sabe qué
        feed, qué tasa ni qué umbrales lo generaron. Se versiona por timestamp
        para conservar el historial de configuraciones.

        Args:
            manifest: Diccionario serializable.
            name: Prefijo del archivo.

        Returns:
            Ruta del manifiesto escrito.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.root / f"{name}_{stamp}.json"
        target.write_text(json.dumps(manifest, indent=2, default=str), "utf-8")
        return target
