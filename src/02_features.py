# %%
# FAZA 2 — Formiranje vremenskih prozora i izvođenje atributa
#
# Iz auth_labeled.parquet (jedan red = jedan autentifikacioni događaj) formiram
# tabelu u kojoj je jedan red = (entitet, vremenski prozor). Za svaki prozor
# računam atribute koji opisuju ponašanje entiteta u tom intervalu, uključujući
# istorijske atribute izvedene isključivo iz prethodnih prozora istog entiteta.
#
# Rezultat je ulaz za modele u Fazi 3.

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


AUTH_LABELED = PROC / "auth_labeled.parquet"

SEC_PER_DAY = 86400
WINDOW_HOURS = 1  # dužina vremenskog prozora u satima
NORMALIZE_ENTITY = True  # True: U10002@DOM1 i U10002@DOM9 su ISTI entitet (U10002)

WINDOW_SEC = WINDOW_HOURS * 3600

# %%
# --- 1. Provera normalizacije entiteta ---
# U skupu se ista osoba javlja pod više naloga: domenski (U10002@DOM1,
# U10002@DOM9) i lokalni (U10004@C21097, U10004@C25763). Ako se svaki string
# tretira kao zaseban entitet, istorija ponašanja te osobe se deli na više
# profila, što slabi atribute. Zato se entitet svodi na identifikator
# korisnika (deo pre znaka @), a opseg naloga (domenski/lokalni) se kasnije
# koristi kao poseban atribut.
con = duckdb.connect()

pregled = con.execute(f"""
    SELECT
        COUNT(DISTINCT dst_user) AS ceo_string,
        COUNT(DISTINCT split_part(dst_user, '@', 1)) AS normalizovano
    FROM read_parquet('{AUTH_LABELED.as_posix()}')
""").df()

print("Različitih entiteta bez normalizacije:", int(pregled["ceo_string"][0]))
print("Različitih entiteta sa normalizacijom:", int(pregled["normalizovano"][0]))


# %%
# --- 2. Agregacija događaja u vremenske prozore ---
# Agregacija se radi u DuckDB-u jer ulazna tabela ima preko 10 miliona redova.
# Prozor se formira samo tamo gde postoji bar jedan događaj, prazni prozori se
# ne generišu, a neaktivnost se kasnije hvata atributom o proteklom vremenu.
# Prozor dobija is_attack = 1 ako sadrži bar jedan događaj označen kao napad

entity_expr = "split_part(dst_user, '@', 1)" if NORMALIZE_ENTITY else "dst_user"

# vremenski_prozori = tabela vremenskih prozora - jedan red = (entitet, prozor)
vremenski_prozori = con.execute(f"""
    SELECT
        {entity_expr}                                    AS entity,
        CAST(time / {WINDOW_SEC} AS BIGINT)              AS window_id,
        COUNT(*)                                         AS n_events,
        SUM(CASE WHEN outcome = 'Success' THEN 1 ELSE 0 END)  AS n_success,
        SUM(CASE WHEN outcome = 'Failure' THEN 1 ELSE 0 END)  AS n_failure,
        COUNT(DISTINCT dst_comp)                         AS n_dst_comp,
        COUNT(DISTINCT src_comp)                         AS n_src_comp,
        SUM(CASE WHEN auth_type = 'NTLM' THEN 1 ELSE 0 END)      AS n_ntlm,
        SUM(CASE WHEN auth_type = 'Kerberos' THEN 1 ELSE 0 END)  AS n_kerberos,
        SUM(CASE WHEN auth_type = '?' THEN 1 ELSE 0 END)         AS n_auth_unknown,
        COUNT(DISTINCT auth_type)                        AS n_auth_types,
        COUNT(DISTINCT logon_type)                       AS n_logon_types,
        SUM(CASE WHEN dst_user LIKE '%@DOM%' THEN 1 ELSE 0 END)  AS n_domain_acct,
        MIN(time)                                        AS t_min,
        MAX(time)                                        AS t_max,
        MAX(is_attack)                                   AS is_attack
    FROM read_parquet('{AUTH_LABELED.as_posix()}')
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()

print(f"Broj prozora ({WINDOW_HOURS}h):", len(vremenski_prozori))
print("Različitih entiteta:", vremenski_prozori["entity"].nunique())
print("\nRaspodela target varijable po prozorima:")
print(vremenski_prozori["is_attack"].value_counts())
print(f"Udeo pozitivnih prozora: {vremenski_prozori['is_attack'].mean() * 100:.4f}%")

# %%
# --- 3. Raspodela napada po danima (osnova za vremensku podelu) ---
# Iz Faze 0 je poznato da su napadi koncentrisani u danima 2-30, a da dani 31-59
# nemaju nijedan napad. Zbog toga podela na trening/validaciju/test mora biti
# postavljena tako da svaki deo sadrži pozitivne primere.
vremenski_prozori["day"] = (
    vremenski_prozori["window_id"] * WINDOW_SEC
) // SEC_PER_DAY + 1

po_danu = vremenski_prozori.groupby("day")["is_attack"].agg(["sum", "count"])
po_danu.columns = ["napadnutih_prozora", "ukupno_prozora"]
print("Prozori sa napadom, po danima (samo dani sa napadima):")
print(po_danu[po_danu["napadnutih_prozora"] > 0])

# %%
# --- Koji redteam dani su izgubljeni zbog CAP_PER_USER ograničenja? ---
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
rt_check = pd.read_csv(
    RAW / "redteam.txt", header=None, names=["time", "user", "src_comp", "dst_comp"]
)
rt_check["day"] = (rt_check["time"] // 86400) + 1
print(rt_check.groupby("day").size())
# %%
# --- 4. Granice vremenske podele ---
# Podela je isključivo vremenska: raniji period služi za obuku, kasniji za
# proveru. Test skup namerno obuhvata i period bez napada (dani 31-59) jer se
# na njemu meri koliko lažnih uzbuna model podiže u mirnom periodu.
TRAIN_END = 12  # trening: dani 1-12
VALID_END = 16  # validacija: dani 13-16; test: dani 17-59


def dodeli_skup(dan: int) -> str:
    if dan <= TRAIN_END:
        return "train"
    if dan <= VALID_END:
        return "valid"
    return "test"


vremenski_prozori["split"] = vremenski_prozori["day"].apply(dodeli_skup)

pregled_podele = vremenski_prozori.groupby("split")["is_attack"].agg(
    ["sum", "count", "mean"]
)
pregled_podele.columns = ["napadnutih_prozora", "ukupno_prozora", "udeo"]
print("Raspodela po skupovima:")
print(pregled_podele)

# %%
# --- 5. Atributi izvedeni iz prozora (bez istorije) ---
vremenski_prozori["fail_ratio"] = (
    vremenski_prozori["n_failure"] / vremenski_prozori["n_events"]
)
vremenski_prozori["ntlm_ratio"] = (
    vremenski_prozori["n_ntlm"] / vremenski_prozori["n_events"]
)
vremenski_prozori["kerberos_ratio"] = (
    vremenski_prozori["n_kerberos"] / vremenski_prozori["n_events"]
)
vremenski_prozori["auth_unknown_ratio"] = (
    vremenski_prozori["n_auth_unknown"] / vremenski_prozori["n_events"]
)
vremenski_prozori["domain_acct_ratio"] = (
    vremenski_prozori["n_domain_acct"] / vremenski_prozori["n_events"]
)

# Prosečan broj događaja po odredišnom računaru — visoka vrednost znači
# koncentrisanu aktivnost, niska znači razuđeno kretanje po više mašina.
vremenski_prozori["events_per_dst"] = (
    vremenski_prozori["n_events"] / vremenski_prozori["n_dst_comp"]
)

# Vremenske odrednice prozora
vremenski_prozori["hour_of_day"] = (
    vremenski_prozori["window_id"] * WINDOW_SEC % SEC_PER_DAY
) // 3600
vremenski_prozori["is_night"] = (
    (vremenski_prozori["hour_of_day"] < 6) | (vremenski_prozori["hour_of_day"] >= 22)
).astype(int)
vremenski_prozori["day_of_week"] = (vremenski_prozori["day"] - 1) % 7
vremenski_prozori["is_weekend"] = (vremenski_prozori["day_of_week"] >= 5).astype(int)

# Rasprostranjenost aktivnosti unutar prozora (u sekundama)
vremenski_prozori["window_duration_sec"] = (
    vremenski_prozori["t_max"] - vremenski_prozori["t_min"]
)

print("Dodati atributi bez istorije. Ukupno kolona:", vremenski_prozori.shape[1])

# %%
# --- 6. Atribut nova_odredista: odredišni računari kojima entitet ranije nije pristupio ---
# Za svaki prozor se računa kom broju odredišnih računara entitet do tada nikada
# nije pristupio. Skup ranije viđenih računara obuhvata isključivo prozore
# PRE tekućeg — tekući prozor se u njega dodaje tek nakon obračuna.
parovi = con.execute(f"""
    SELECT DISTINCT
        {entity_expr}                        AS entity,
        CAST(time / {WINDOW_SEC} AS BIGINT)  AS window_id,
        dst_comp
    FROM read_parquet('{AUTH_LABELED.as_posix()}')
    ORDER BY 1, 2
""").df()

novi_po_prozoru = []
for korisnik, grupa in parovi.groupby("entity", sort=False):
    vidjeni = set()
    for window_id, pod in grupa.groupby("window_id", sort=True):
        trenutni = set(pod["dst_comp"])
        novi = len(trenutni - vidjeni)
        novi_po_prozoru.append((korisnik, window_id, novi, len(vidjeni)))
        vidjeni |= trenutni

nova_odredista = pd.DataFrame(
    novi_po_prozoru, columns=["entity", "window_id", "n_new_dst", "n_dst_seen_before"]
)
vremenski_prozori = vremenski_prozori.merge(
    nova_odredista, on=["entity", "window_id"], how="left"
)
vremenski_prozori["new_dst_ratio"] = (
    vremenski_prozori["n_new_dst"] / vremenski_prozori["n_dst_comp"]
)

print("Dodati atributi nova_odredista.")
print(
    "Prosečan broj novih odredišnih računara po prozoru:",
    round(vremenski_prozori["n_new_dst"].mean(), 3),
)

# %%
# --- 7. Istorijski atributi (isključivo iz prethodnih prozora) ---
# Svaki istorijski atribut se računa nad pomerenom serijom (shift(1)), čime se
# tekući prozor izostavlja iz sopstvene istorije. Bez tog pomeranja model bi
# posredno video vrednost koju treba da predvidi.
vremenski_prozori = vremenski_prozori.sort_values(["entity", "window_id"]).reset_index(
    drop=True
)

HISTORY_COLUMNS = ["n_events", "n_dst_comp", "n_src_comp", "n_failure", "n_ntlm"]

g = vremenski_prozori.groupby("entity", sort=False)

# Vreme proteklo od prethodne aktivnosti (u satima)
vremenski_prozori["prev_window_id"] = g["window_id"].shift(1)
vremenski_prozori["hours_since_prev"] = (
    vremenski_prozori["window_id"] - vremenski_prozori["prev_window_id"]
) * WINDOW_HOURS
vremenski_prozori["hours_since_prev"] = vremenski_prozori["hours_since_prev"].fillna(-1)

# Redni broj prozora za entitet - koliko istorije uopšte postoji
vremenski_prozori["window_seq_num"] = g.cumcount()

for kol in HISTORY_COLUMNS:
    pomeren = g[kol].shift(1)
    vremenski_prozori[f"{kol}_lag1"] = pomeren

    # Pokretni prosek i standardna devijacija nad prethodnih 24 prozora
    vremenski_prozori[f"{kol}_ma24"] = pomeren.groupby(
        vremenski_prozori["entity"]
    ).transform(lambda s: s.rolling(24, min_periods=2).mean())
    vremenski_prozori[f"{kol}_sd24"] = pomeren.groupby(
        vremenski_prozori["entity"]
    ).transform(lambda s: s.rolling(24, min_periods=2).std())

    # Odstupanje tekućeg prozora od sopstvenog proseka entiteta (z-skor)
    vremenski_prozori[f"{kol}_z"] = (
        vremenski_prozori[kol] - vremenski_prozori[f"{kol}_ma24"]
    ) / vremenski_prozori[f"{kol}_sd24"].replace(0, np.nan)

    # Promena u odnosu na prethodni prozor
    vremenski_prozori[f"{kol}_delta"] = (
        vremenski_prozori[kol] - vremenski_prozori[f"{kol}_lag1"]
    )

vremenski_prozori = vremenski_prozori.drop(columns=["prev_window_id"])
print("Dodati istorijski atributi. Ukupno kolona:", vremenski_prozori.shape[1])

# %%
# --- 8. Kontrola curenja podataka ---
# Provera da nijedan istorijski atribut nije izračunat iz tekućeg prozora:
# za prvi prozor svakog entiteta sve lag vrednosti moraju biti nedostajuće.
prvi = vremenski_prozori[vremenski_prozori["window_seq_num"] == 0]
lag_kolone = [c for c in vremenski_prozori.columns if c.endswith("_lag1")]
provera = prvi[lag_kolone].notna().sum().sum()

print("Broj popunjenih lag vrednosti u prvom prozoru entiteta:", provera)
assert provera == 0, "Istorijski atribut je izračunat iz tekućeg prozora."

# Provera da se skupovi vremenski ne preklapaju
granice = vremenski_prozori.groupby("split")["day"].agg(["min", "max"])
print("\nVremenski opseg po skupovima:")
print(granice)

# %%
# --- 9. Čuvanje tabele atributa ---
OUTPUT = PROC / "features.parquet"
vremenski_prozori.to_parquet(OUTPUT, index=False)

print("Sačuvano:", OUTPUT.name)
print("Dimenzije tabele:", vremenski_prozori.shape)
print("\nRaspodela po skupovima i target varijabli:")
print(pd.crosstab(vremenski_prozori["split"], vremenski_prozori["is_attack"]))
# %%
