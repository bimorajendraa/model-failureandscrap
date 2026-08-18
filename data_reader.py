"""Akses database READ-ONLY.

Modul ini membangun ulang rantai pembersihan data dari TABEL MENTAH
(journal / inventory / master) memakai query SELECT saja, sehingga
production_ml tidak bergantung sama sekali pada schema `analytics` hasil
research dan tidak pernah membuat object apa pun di database.

Rantai yang dibangun ulang (mengikuti logic research yang sudah terbukti):

    journal.t_item_journey  ->  event bersih (kode dinormalisasi, client &
                                lokasi dikanonikalisasi)
                            ->  event operasional (buang RECON administratif
                                dan tanggal tidak valid)
                            ->  failure event (DISMANTLED + CORRECTIVE, atau
                                PREVENTIVE yang berakhir BROKEN/UNREPAIRABLE)
                            ->  installation cycle (satu baris per pemasangan)

Tanggung jawab modul ini berhenti di situ: MEMBACA. Pembentukan observasi,
agregat riwayat, dan fitur dikerjakan feature_builder.py.

Yang SENGAJA tidak dibawa dari research: seluruh kolom yang tidak dipakai 18
fitur final (relocation/preventive/repair-process counts, window yang tidak
terpakai, last_status, last_place, hierarchy TERMINAL, kolom audit kualitas
data). Lihat README.md.
"""

from __future__ import annotations

import psycopg
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Koneksi
# ---------------------------------------------------------------------------


def connect() -> psycopg.Connection:
    """Koneksi read-only. Sesi dipaksa read-only di level database, jadi query
    apa pun yang mencoba menulis akan ditolak PostgreSQL."""
    return psycopg.connect(
        **config.db_settings(),
        application_name="production_ml",
        connect_timeout=10,
        options="-c default_transaction_read_only=on",
    )


def _query(conn: psycopg.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


# ---------------------------------------------------------------------------
# Kanonikalisasi client & lokasi
#
# Nilai mentah dipetakan ke nama master lewat tiga tahap berurutan: cocok
# persis -> alias yang sudah disetujui -> fuzzy match yang sangat mirip. Tahap
# fuzzy penting, bukan kosmetik: 31% baris journey menuliskan nama client
# dengan typo ("KERETE COMMUTER INDONESIA"), dan tanpa tahap itu fitur
# client_category baris-baris tersebut akan jatuh ke UNKNOWN.
# ---------------------------------------------------------------------------


def _clean(column: str) -> str:
    """Ekspresi SQL normalisasi kode/nama: trim, rapatkan spasi, UPPER."""
    return f"NULLIF(UPPER(REGEXP_REPLACE(TRIM({column}), '\\s+', ' ', 'g')), '')"


def _recon_context(
    wo_type: str, work_order_type: str, status: str, activity: str,
    done_by: str, remark_upper: str,
) -> str:
    """Apakah event ini sekadar pembukuan RECON, bukan kejadian teknis.

    Satu-satunya definisi RECON di modul ini - dipakai rantai utama maupun
    query batas tanggal, supaya keduanya tidak mungkin berbeda aturan.
    """
    return f"""(
        COALESCE({wo_type} = 'RECON', FALSE)
        OR COALESCE({work_order_type} = 'RECON', FALSE)
        OR COALESCE({status} = 'DISMANTLED' AND {activity} = 'RECON', FALSE)
        OR COALESCE(
            {status} = 'INSTALLED'
            AND (POSITION('RECON' IN COALESCE({done_by}, '')) > 0
                 OR POSITION('RECON' IN {remark_upper}) > 0),
            FALSE
        )
    )"""


def _valid_operational_date(created_on: str) -> str:
    """Tanggal yang masuk akal dipakai: bukan kosong, bukan sebelum sistem
    pencatatan ada, dan bukan tanggal masa depan (salah input)."""
    return f"""(
        {created_on} IS NOT NULL
        AND {created_on}::date >= DATE '1971-01-01'
        AND {created_on} <= CURRENT_TIMESTAMP
    )"""


def _inventory_lookup_cte() -> str:
    """Identitas PART menurut inventory, dipakai memastikan model PART konsisten.

    Dipakai bersama oleh pembacaan siklus dan pembacaan episode kerusakan -
    ditulis sekali di sini supaya keduanya tidak mungkin memakai aturan
    identitas yang berbeda.
    """
    return f"""inventory_identifier AS MATERIALIZED (
    SELECT DISTINCT
        {_clean('i.item_pairing_code')} AS item_pairing_code_clean,
        {_clean('i.sn_ref')} AS sn_ref_clean,
        CASE WHEN {_clean('i.item_model_code')} IS NOT NULL
              AND {_clean('i.item_pairing_code')} IS NOT NULL
              AND {_clean('i.repair_seq')} IS NOT NULL
             THEN {_clean("i.item_model_code || '-' || i.item_pairing_code || '-' || i.repair_seq")}
        END AS host_serial_code_clean,
        {_clean('i.item_model_code')} AS item_model_code_clean
    FROM inventory.t_item i
),

inventory_lookup AS MATERIALIZED (
    SELECT lookup_type, identifier_clean,
        COUNT(DISTINCT item_model_code_clean) AS nonnull_model_count,
        BOOL_OR(item_model_code_clean IS NULL) AS has_null_model,
        MIN(item_model_code_clean) AS only_model_code
    FROM (
        SELECT 'PAIRING'::text AS lookup_type,
            item_pairing_code_clean AS identifier_clean, item_model_code_clean
        FROM inventory_identifier WHERE item_pairing_code_clean IS NOT NULL
        UNION ALL
        SELECT 'HOST', host_serial_code_clean, item_model_code_clean
        FROM inventory_identifier WHERE host_serial_code_clean IS NOT NULL
        UNION ALL
        SELECT 'HOST', sn_ref_clean, item_model_code_clean
        FROM inventory_identifier WHERE sn_ref_clean IS NOT NULL
    ) v
    GROUP BY lookup_type, identifier_clean
)"""


def _matches_inventory(pairing: str, host: str, model_column: str) -> str:
    """PART dikenali inventory DAN model teknisnya tidak bertentangan."""
    return f"""(
        ({pairing}.identifier_clean IS NOT NULL OR {host}.identifier_clean IS NOT NULL)
        AND CASE WHEN {pairing}.identifier_clean IS NULL THEN TRUE
                 WHEN {model_column} IS NULL THEN {pairing}.nonnull_model_count = 0
                 ELSE NOT {pairing}.has_null_model AND {pairing}.nonnull_model_count = 1
                      AND {pairing}.only_model_code = {model_column} END
        AND CASE WHEN {host}.identifier_clean IS NULL THEN TRUE
                 WHEN {model_column} IS NULL THEN {host}.nonnull_model_count = 0
                 ELSE NOT {host}.has_null_model AND {host}.nonnull_model_count = 1
                      AND {host}.only_model_code = {model_column} END
    )"""


def _work_order_type_cte() -> str:
    """Jenis pekerjaan dari work order, dipakai mendeteksi konteks RECON."""
    return f"""work_order_type AS MATERIALIZED (
    SELECT DISTINCT ON ({_clean('wo.wo_code')})
        {_clean('wo.wo_code')} AS wo_code_clean,
        COALESCE({_clean('mwt.work_type')}, {_clean('wo.work_type_code')})
            AS work_order_type_clean
    FROM journal.t_work_order wo
    LEFT JOIN master.t_mtr_work_type mwt
        ON {_clean('mwt.work_type_code')} = {_clean('wo.work_type_code')}
        OR {_clean('mwt.work_type')} = {_clean('wo.work_type_code')}
    WHERE {_clean('wo.wo_code')} IS NOT NULL
    ORDER BY {_clean('wo.wo_code')}, wo.wo_id DESC
)"""


def _fuzzy_key(value: str) -> str:
    """Kunci pembanding: tanda baca disamakan jadi spasi, singkatan umum yang
    tidak ambigu diperluas."""
    words = "".join(ch if ch.isalnum() else " " for ch in value.upper()).split()
    return " ".join(config.TEXT_ABBREVIATION_MAPPING.get(w, w) for w in words)


def _levenshtein(left: str, right: str) -> int:
    if not left or not right:
        return len(left) or len(right)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _similarity(left: str, right: str) -> float:
    left_key, right_key = _fuzzy_key(left), _fuzzy_key(right)
    longest = max(len(left_key), len(right_key))
    if longest == 0:
        return 0.0
    return round(1.0 - _levenshtein(left_key, right_key) / longest, 4)


def _canonical_map(
    source_values: list[str],
    master_pairs: list[tuple[str | None, str]],
    approved_alias: dict[str, str],
) -> dict[str, str]:
    """Petakan setiap nilai sumber ke satu nama master.

    `master_pairs` adalah pasangan (kode, nama) dari tabel master; kode maupun
    nama sama-sama bisa dicocokkan persis. Nilai yang tetap tidak terpetakan
    sengaja dibiarkan absen dari hasil (nanti jadi NULL -> UNKNOWN), bukan
    ditebak.
    """
    master_names = sorted({name for _, name in master_pairs if name})

    # Satu nilai sumber bisa cocok persis dengan >1 nama master (ambigu);
    # kasus itu tidak boleh dipetakan otomatis.
    exact: dict[str, set[str]] = {}
    for code, name in master_pairs:
        for key in (code, name):
            if key:
                exact.setdefault(key, set()).add(name)

    alias = {
        source.upper(): canonical
        for source, canonical in approved_alias.items()
        if canonical in master_names
    }

    mapping: dict[str, str] = {}
    for source in source_values:
        if source in exact:
            names = exact[source]
            if len(names) == 1:
                mapping[source] = next(iter(names))
            elif source in alias:  # cocok persis tapi ambigu -> hanya alias
                mapping[source] = alias[source]
            continue
        if source in alias:
            mapping[source] = alias[source]
            continue

        scored = sorted(
            ((_similarity(source, name), name) for name in master_names),
            key=lambda item: (-item[0], item[1]),
        )
        if not scored:
            continue
        best_score, best_name = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if (
            best_score >= config.FUZZY_MIN_SCORE
            and best_score - second_score >= config.FUZZY_MIN_MARGIN
        ):
            mapping[source] = best_name
    return mapping


_TEXT_MAPS: tuple[dict[str, str], dict[str, str]] | None = None


def _build_text_maps(conn: psycopg.Connection) -> tuple[dict[str, str], dict[str, str]]:
    """Bangun mapping client & lokasi, sekali per proses.

    Jumlah nilai unik kecil (~5 client, ~156 lokasi) sehingga murah dihitung
    penuh, tetapi hasilnya dipakai berulang oleh beberapa query.
    """
    global _TEXT_MAPS
    if _TEXT_MAPS is not None:
        return _TEXT_MAPS

    clients = _query(
        conn,
        f"""
        SELECT DISTINCT {_clean('j.client')} AS value FROM journal.t_item_journey j
        WHERE {_clean('j.client')} IS NOT NULL
        """,
    )["value"].tolist()
    master_clients = _query(
        conn,
        f"""
        SELECT DISTINCT {_clean('c.client_code')} AS code, {_clean('c.client_name')} AS name
        FROM master.t_mtr_client c WHERE {_clean('c.client_name')} IS NOT NULL
        """,
    )
    places = _query(
        conn,
        f"""
        SELECT DISTINCT {_clean('j.place')} AS value FROM journal.t_item_journey j
        WHERE {_clean('j.place')} IS NOT NULL
        """,
    )["value"].tolist()
    master_places = _query(
        conn,
        f"""
        SELECT DISTINCT {_clean('l.location_code')} AS code, {_clean('l.location_name')} AS name
        FROM master.t_mtr_location l WHERE {_clean('l.location_name')} IS NOT NULL
        """,
    )

    client_map = _canonical_map(
        clients,
        list(master_clients.itertuples(index=False, name=None)),
        config.APPROVED_CLIENT_ALIAS,
    )
    place_map = _canonical_map(
        places,
        list(master_places.itertuples(index=False, name=None)),
        config.APPROVED_LOCATION_ALIAS,
    )
    _TEXT_MAPS = (client_map, place_map)
    return _TEXT_MAPS


def _values_cte(name: str, mapping: dict[str, str]) -> str:
    """Mapping Python -> CTE VALUES supaya join tetap dikerjakan database."""
    if not mapping:
        return (
            f"{name}(source_value, canonical_value) AS "
            "(SELECT NULL::text, NULL::text WHERE FALSE)"
        )
    quote = "'"
    rows = ", ".join(
        "(" + ", ".join(f"{quote}{v.replace(quote, quote * 2)}{quote}" for v in pair) + ")"
        for pair in mapping.items()
    )
    return f"{name}(source_value, canonical_value) AS (VALUES {rows})"


# ---------------------------------------------------------------------------
# Rantai SQL bersama
# ---------------------------------------------------------------------------


def _chain_sql(
    client_map: dict[str, str],
    place_map: dict[str, str],
    *,
    single_item: bool,
    with_failures: bool = True,
) -> str:
    """CTE dari tabel mentah sampai `operational` (event siap pakai).

    with_failures=False melewati penentuan failure onset. Dipakai query yang
    memang tidak memerlukannya (mis. mencari tanggal data terbaru), karena
    tahap itu jauh paling mahal di seluruh rantai.
    """
    # Filter memakai identifier turunan yang sama persis dengan yang dipakai
    # seluruh rantai, bukan kolom mentah - kalau memfilter host_serial_code
    # mentah, baris milik item lain yang identifier-nya diambil dari
    # item_pairing_code bisa ikut terbawa.
    item_filter = (
        f"""WHERE COALESCE(
            {_clean('j.item_pairing_code')},
            {_clean('j.host_serial_code')},
            'JOURNEY#' || j.journey_id::text
        ) = %s"""
        if single_item
        else ""
    )

    failure_cte = _FAILURE_CTE if with_failures else ""
    failure_flag = "f.journey_id IS NOT NULL" if with_failures else "FALSE"
    failure_join = (
        "LEFT JOIN failure_event f ON f.journey_id = e.journey_id"
        if with_failures
        else ""
    )

    return f"""
WITH
{_values_cte('client_map', client_map)},
{_values_cte('place_map', place_map)},

-- Event journey yang sudah dinormalisasi dan dikanonikalisasi.
event AS MATERIALIZED (
    SELECT
        j.journey_id,
        {_clean('j.item_category')} AS item_category_clean,
        {_clean('j.item_type')} AS item_type_clean,
        {_clean('j.item_model_code')} AS item_model_code_clean,
        {_clean('j.item_pairing_code')} AS item_pairing_code_clean,
        {_clean('j.host_serial_code')} AS host_serial_code_clean,
        COALESCE(
            {_clean('j.item_pairing_code')},
            {_clean('j.host_serial_code')},
            'JOURNEY#' || j.journey_id::text
        ) AS item_identifier_clean,
        cm.canonical_value AS client_clean,
        pm.canonical_value AS place_canonical_clean,
        {_clean('j.wo_type')} AS wo_type_clean,
        {_clean('j.wo_code')} AS wo_code_clean,
        {_clean('j.activity')} AS activity_clean,
        {_clean('j.status')} AS status_clean,
        {_clean('j.done_by')} AS done_by_clean,
        UPPER(COALESCE(j.remark, '')) AS remark_upper,
        j.created_on
    FROM journal.t_item_journey j
    LEFT JOIN client_map cm ON cm.source_value = {_clean('j.client')}
    LEFT JOIN place_map pm ON pm.source_value = {_clean('j.place')}
    {item_filter}
),

{_work_order_type_cte()},

{failure_cte}
-- RECON administratif tidak mencerminkan kondisi teknis PART, jadi tidak
-- boleh ikut menghitung durasi/urutan operasional.
semantic AS MATERIALIZED (
    SELECT e.*,
        {failure_flag} AS is_failure_onset,
        {_recon_context(
            "e.wo_type_clean", "w.work_order_type_clean", "e.status_clean",
            "e.activity_clean", "e.done_by_clean", "e.remark_upper",
        )} AS is_admin_recon_context,
        {_valid_operational_date("e.created_on")} AS is_valid_operational_date
    FROM event e
    LEFT JOIN work_order_type w ON w.wo_code_clean = e.wo_code_clean
    {failure_join}
),

operational AS MATERIALIZED (
    SELECT * FROM semantic
    WHERE is_valid_operational_date AND NOT is_admin_recon_context
)
"""


# Failure onset ada dua dasar. Dasar pertama: pembongkaran korektif, yang
# dengan sendirinya sudah berarti PART bermasalah.
_FAILURE_CTE = """
failure_event AS MATERIALIZED (
    SELECT journey_id, item_identifier_clean, created_on AS failure_onset_on
    FROM event
    WHERE status_clean = 'DISMANTLED' AND wo_type_clean = 'CORRECTIVE'

    UNION ALL

    -- Dasar kedua: pembongkaran preventif yang TERNYATA berakhir rusak
    -- sebelum PART itu dipasang lagi. Jumlahnya sedikit, tapi mengabaikannya
    -- berarti melabeli kerusakan nyata sebagai bukan-kerusakan.
    SELECT c.journey_id, c.item_identifier_clean, c.created_on
    FROM (
        SELECT * FROM event
        WHERE status_clean = 'DISMANTLED' AND wo_type_clean = 'PREVENTIVE'
    ) c
    LEFT JOIN LATERAL (
        SELECT o.created_on, o.journey_id
        FROM event o
        WHERE o.item_identifier_clean = c.item_identifier_clean
          AND o.status_clean IN ('UNREPAIRABLE', 'BROKEN', 'SENDLOG (BROKEN)')
          AND (o.created_on, o.journey_id) > (c.created_on, c.journey_id)
        ORDER BY o.created_on, o.journey_id LIMIT 1
    ) broken ON TRUE
    LEFT JOIN LATERAL (
        SELECT n.created_on, n.journey_id
        FROM event n
        WHERE n.item_identifier_clean = c.item_identifier_clean
          AND n.status_clean = 'INSTALLED'
          AND (n.created_on, n.journey_id) > (c.created_on, c.journey_id)
        ORDER BY n.created_on, n.journey_id LIMIT 1
    ) reinstall ON TRUE
    WHERE broken.created_on IS NOT NULL
      AND (
          reinstall.created_on IS NULL
          OR (broken.created_on, broken.journey_id)
                 < (reinstall.created_on, reinstall.journey_id)
      )
),
"""


# ---------------------------------------------------------------------------
# Pembacaan
# ---------------------------------------------------------------------------


def get_dataset_max_event_on() -> pd.Timestamp:
    """Kejadian terbaru yang tercatat di database. Ini batas waktu "sekarang"
    menurut data, dan sengaja dipakai sebagai titik observasi prediction."""
    # Query ini sengaja tidak memakai rantai penuh: mencari tanggal terbaru
    # tidak butuh kanonikalisasi teks maupun penentuan failure, dan
    # melewatinya membuat prediksi satu PART jauh lebih cepat. Aturan RECON
    # dan tanggal valid tetap memakai definisi yang sama persis.
    sql = f"""
WITH {_work_order_type_cte()}
SELECT MAX(j.created_on) AS dataset_max_event_on
FROM journal.t_item_journey j
LEFT JOIN work_order_type w ON w.wo_code_clean = {_clean('j.wo_code')}
WHERE {_valid_operational_date('j.created_on')}
  AND NOT {_recon_context(
      _clean('j.wo_type'), "w.work_order_type_clean", _clean('j.status'),
      _clean('j.activity'), _clean('j.done_by'), "UPPER(COALESCE(j.remark, ''))",
  )}
"""
    with connect() as conn:
        return pd.Timestamp(_query(conn, sql).iloc[0, 0])


def get_events(item_id: str | None = None) -> pd.DataFrame:
    """Event operasional yang dipakai menghitung riwayat point-in-time."""
    with connect() as conn:
        client_map, place_map = _build_text_maps(conn)
        sql = (
            _chain_sql(client_map, place_map, single_item=item_id is not None)
            + """
SELECT journey_id, item_identifier_clean, created_on, wo_type_clean, status_clean,
       item_type_clean, is_failure_onset, place_canonical_clean
FROM operational
WHERE item_identifier_clean IS NOT NULL
ORDER BY item_identifier_clean, created_on, journey_id
"""
        )
        return _query(conn, sql, () if item_id is None else (_normalize(item_id),))


def get_cycles(
    item_id: str | None = None, dataset_max_event_on: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Satu baris per pemasangan PART, lengkap dengan cara siklus itu berakhir.

    `dataset_max_event_on` wajib diisi kalau `item_id` diisi: batas waktu data
    bersifat global, tidak boleh dihitung ulang dari riwayat satu item saja.
    """
    if item_id is not None and dataset_max_event_on is None:
        raise ValueError("dataset_max_event_on wajib diisi saat membaca satu item.")

    boundary = (
        "SELECT %s::timestamp AS dataset_max_event_on"
        if item_id is not None
        else "SELECT MAX(created_on) AS dataset_max_event_on FROM operational"
    )
    horizon = f"INTERVAL '{config.TARGET_HORIZON_DAYS} days'"

    with connect() as conn:
        client_map, place_map = _build_text_maps(conn)
        sql = (
            _chain_sql(client_map, place_map, single_item=item_id is not None)
            + f"""
, dataset_boundary AS ({boundary}),

item_boundary AS (
    SELECT item_identifier_clean, MAX(created_on) AS item_last_seen_on
    FROM operational WHERE item_identifier_clean IS NOT NULL
    GROUP BY item_identifier_clean
),

-- RECON yang muncul SETELAH item terakhir terlihat aktif menandakan ada yang
-- perlu direkonsiliasi: masa diam sebelumnya belum tentu benar-benar aman.
recon_after AS (
    SELECT b.item_identifier_clean,
        BOOL_OR(s.created_on > b.item_last_seen_on) AS has_recon_after_last_seen
    FROM item_boundary b
    JOIN semantic s
      ON s.item_identifier_clean = b.item_identifier_clean AND s.is_admin_recon_context
    GROUP BY b.item_identifier_clean
),

{_inventory_lookup_cte()},

-- Siklus hanya dibuka oleh pemasangan (INSTALLED) PART.
installed_event AS MATERIALIZED (
    SELECT o.*,
        ROW_NUMBER() OVER install_order AS installation_sequence,
        LEAD(o.created_on) OVER install_order AS next_installed_on
    FROM operational o
    WHERE o.status_clean = 'INSTALLED'
      AND o.item_category_clean = 'PART'
      AND o.item_identifier_clean IS NOT NULL
    WINDOW install_order AS (
        PARTITION BY o.item_identifier_clean ORDER BY o.created_on, o.journey_id
    )
),

-- Setiap failure dipasangkan ke siklus yang sedang berjalan saat itu dengan
-- menghitung berapa pemasangan yang sudah terjadi sampai titik tersebut.
-- Ini setara dengan mencari failure pertama di antara dua pemasangan, tetapi
-- tidak perlu membandingkan setiap failure dengan setiap pemasangan.
event_stream AS (
    SELECT item_identifier_clean, created_on, journey_id, 1 AS is_install
    FROM installed_event
    UNION ALL
    SELECT item_identifier_clean, failure_onset_on, journey_id, 0 FROM failure_event
),
failure_in_cycle AS MATERIALIZED (
    SELECT DISTINCT ON (item_identifier_clean, installation_sequence)
        item_identifier_clean, installation_sequence, created_on AS failure_onset_on
    FROM (
        SELECT *, SUM(is_install) OVER (
            PARTITION BY item_identifier_clean ORDER BY created_on, journey_id
            ROWS UNBOUNDED PRECEDING
        ) AS installation_sequence
        FROM event_stream
    ) tagged
    WHERE is_install = 0 AND installation_sequence > 0
    ORDER BY item_identifier_clean, installation_sequence, created_on, journey_id
),

cycle_base AS (
    SELECT
        i.item_identifier_clean || ':' || i.installation_sequence::text
            AS installation_cycle_id,
        i.item_identifier_clean,
        i.installation_sequence,
        i.created_on AS installed_on,
        i.item_model_code_clean,
        i.client_clean AS installed_client_clean,
        f.failure_onset_on,
        i.next_installed_on,
        b.dataset_max_event_on,
        COALESCE(f.failure_onset_on, i.next_installed_on, b.dataset_max_event_on)
            AS cycle_end_on,
        CASE WHEN f.failure_onset_on IS NOT NULL THEN 'FAILURE'
             WHEN i.next_installed_on IS NOT NULL THEN 'REINSTALL_WITHOUT_RECORDED_FAILURE'
             ELSE 'RIGHT_CENSORED_AT_DATA_END' END AS cycle_end_reason,
        -- Snapshot boleh jadi negatif kalau siklusnya berakhir karena failure
        -- (yang bisa saja di luar horizon) atau masih berjalan TANPA jejak
        -- RECON belakangan. Siklus yang ditutup pemasangan ulang tanpa failure
        -- tercatat tidak boleh otomatis dianggap negatif - tidak ada yang tahu
        -- apa yang sebenarnya terjadi pada PART itu.
        (
            f.failure_onset_on IS NOT NULL
            OR (
                i.next_installed_on IS NULL
                AND NOT COALESCE(ra.has_recon_after_last_seen, FALSE)
            )
        ) AS is_recon_verified_negative_eligible,
        (
            {_matches_inventory("pl", "hl", "i.item_model_code_clean")}
            AND i.created_on < COALESCE(
                f.failure_onset_on, i.next_installed_on, b.dataset_max_event_on
            )
        ) AS is_initial_model_cohort
    FROM installed_event i
    CROSS JOIN dataset_boundary b
    LEFT JOIN recon_after ra ON ra.item_identifier_clean = i.item_identifier_clean
    LEFT JOIN failure_in_cycle f
        ON f.item_identifier_clean = i.item_identifier_clean
       AND f.installation_sequence = i.installation_sequence
    LEFT JOIN inventory_lookup pl
        ON pl.lookup_type = 'PAIRING' AND pl.identifier_clean = i.item_pairing_code_clean
    LEFT JOIN inventory_lookup hl
        ON hl.lookup_type = 'HOST' AND hl.identifier_clean = i.host_serial_code_clean
)

-- Rata-rata umur siklus SEBELUMNYA untuk item yang sama (bukan siklus ini).
SELECT c.installation_cycle_id, c.item_identifier_clean, c.installed_on,
    c.item_model_code_clean, c.installed_client_clean,
    c.failure_onset_on, c.cycle_end_on, c.cycle_end_reason,
    c.dataset_max_event_on,
    c.is_recon_verified_negative_eligible, c.is_initial_model_cohort,
    -- Batas akhir observasi yang hasil negatifnya masih bisa dipastikan.
    LEAST(c.cycle_end_on, c.dataset_max_event_on) - {horizon} AS last_confirmable_observation_on,
    AVG(EXTRACT(EPOCH FROM (c.cycle_end_on - c.installed_on)) / 86400.0) OVER previous_cycles
        AS previous_cycle_lifetime_mean,
    COUNT(*) OVER previous_cycles > 0 AS has_previous_cycle
FROM cycle_base c
WINDOW previous_cycles AS (
    PARTITION BY c.item_identifier_clean ORDER BY c.installation_sequence
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
)
ORDER BY c.item_identifier_clean, c.installation_sequence
"""
        )
        params: tuple = ()
        if item_id is not None:
            # Urutan mengikuti kemunculan placeholder: filter item di CTE
            # `event`, lalu batas tanggal data di `dataset_boundary`.
            params = (_normalize(item_id), pd.Timestamp(dataset_max_event_on).to_pydatetime())
        return _query(conn, sql, params)


def get_failure_episodes(item_id: str | None = None) -> pd.DataFrame:
    """Setiap kejadian kerusakan PART, sebagai bahan model risiko scrap.

    Satu baris = satu kerusakan. Berbeda dari get_cycles(): sebuah siklus
    pemasangan hanya mencatat kerusakan PERTAMA yang mengakhirinya, sedangkan
    di sini semua kerusakan ikut - termasuk yang terjadi sebelum pemasangan
    pertama tercatat, yang jumlahnya tidak sedikit.
    """
    with connect() as conn:
        client_map, place_map = _build_text_maps(conn)
        sql = (
            _chain_sql(client_map, place_map, single_item=item_id is not None)
            + f"""
, {_inventory_lookup_cte()}

SELECT
    f.journey_id AS onset_journey_id,
    f.item_identifier_clean,
    f.failure_onset_on,
    e.item_type_clean,
    e.item_model_code_clean,
    {_matches_inventory("pl", "hl", "e.item_model_code_clean")}
        AND e.item_category_clean = 'PART'
        AS is_initial_model_cohort
FROM failure_event f
JOIN event e ON e.journey_id = f.journey_id
LEFT JOIN inventory_lookup pl
    ON pl.lookup_type = 'PAIRING' AND pl.identifier_clean = e.item_pairing_code_clean
LEFT JOIN inventory_lookup hl
    ON hl.lookup_type = 'HOST' AND hl.identifier_clean = e.host_serial_code_clean
ORDER BY f.item_identifier_clean, f.failure_onset_on, f.journey_id
"""
        )
        params = () if item_id is None else (_normalize(item_id),)
        return _query(conn, sql, params)


def _normalize(item_id: str) -> str:
    """Samakan bentuk identifier dengan normalisasi yang dipakai di SQL."""
    return " ".join(str(item_id).strip().upper().split())
