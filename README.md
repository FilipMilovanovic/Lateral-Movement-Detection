# Master rad

Implementacija master rada **„Otkrivanje lateralnog kretanja na osnovu autentifikacionih događaja primenom vremenskih serija i objašnjivog mašinskog učenja"**.

- **Autor:** Filip Milovanović
- **Mentor:** doc. dr Sandro Radovanović

## O projektu

Lateralno kretanje je faza napada u kojoj napadač, nakon inicijalnog upada, koristi kompromitovane autentifikacione podatke da pristupi drugim računarima u mreži.

Projekat modeluje autentifikacione događaje iz Windows/Active Directory okruženja kao vremenske serije po entitetu i poredi tri pristupa detekciji:

- **klasične modele** (bazni model, logistička regresija, stablo odlučivanja, KNN, slučajna šuma, XGBoost),
- **duboke sekvencijalne modele** (LSTM, GRU, Transformer),
- **nenadgledanu detekciju** (Isolation Forest, jednoklasna mašina sa potpornim vektorima).

Najbolji model je zatim objašnjen pomoću SHAP metode, a najuticajniji atributi su mapirani na taktike i tehnike iz MITRE ATT&CK matrice.

## Glavni rezultati

Osnovna mera je prosečna preciznost (AP), jer tačnost i AUC-ROC obmanjuju pri ekstremnoj neravnoteži klasa (napadi čine 0,65% prozora na validaciji i 0,02% na testu).

**Validacioni skup (dani 13–16):**

| Pristup | Najbolja konfiguracija | AP |
|---|---|---|
| Slučajna šuma | Filter (40 atributa), bez balansiranja | 0,2332 |
| XGBoost | Filter (40 atributa) | 0,1868 |
| GRU | Filter (40 atributa) | 0,1311 |
| Isolation Forest | Filter (40 atributa) | 0,0598 |
| Bazni model (slučajno pogađanje) | | 0,0065 |

**Test skup (dani 17–59, korišćen jednom, na kraju istraživanja):**

| Model | AP | 95% interval poverenja | AUC-ROC | Detektovano napada | Lažne uzbune dnevno |
|---|---|---|---|---|---|
| Slučajna šuma | 0,0286 | [0,0049; 0,1195] | 0,9431 | 10 od 35 | 12,9 |
| GRU | 0,0015 | [0,0008; 0,0035] | 0,8088 | 4 od 35 | 32,7 |
| Isolation Forest | 0,0020 | [0,0010; 0,0046] | 0,8813 | 7 od 35 | 74,1 |

SHAP analiza pokazuje da se model najviše oslanja na atribute NTLM protokola (udeo i nagle promene u odnosu na uobičajeno ponašanje naloga), što odgovara tehnici Pass-the-Hash (T1550.002), kao i na pristup ranije neposećenim odredišnim računarima, što odgovara taktici Lateral Movement (TA0008).

## Skup podataka

Korišćen je javno dostupan skup **Comprehensive, Multi-Source Cyber-Security Events** Nacionalne laboratorije Los Alamos (Kent, 2015):

https://csr.lanl.gov/data/cyber1/

Podaci nisu deo repozitorijuma zbog veličine. Za pokretanje je potrebno preuzeti fajlove `auth.txt` i `redteam.txt`, raspakovati ih i smestiti u:

```
data/raw/auth.txt
data/raw/redteam.txt
```

Iz skupa je formiran uzorak od 504 naloga (svih 104 naloga iz redteam evidencije i 400 nasumično izabranih kontrolnih naloga), koji se nakon normalizacije svode na 460 entiteta i 224.180 jednosatnih prozora.

## Struktura repozitorijuma

```
├── data/
│   ├── raw/              # izvorni LANL fajlovi (nisu u repozitorijumu)
│   └── processed/        # međurezultati koje generišu skripte
├── models/               # sačuvani modeli i njihove konfiguracije
├── notebooks/            # sveske sa ispisima i grafikonima
├── results/              # tabele (CSV) i grafikoni (PNG)
├── src/                  # skripte
└── requirements.txt	  
```

Skripte u folderu `src/` i sveske u folderu `notebooks/` imaju isti sadržaj. Skripte su organizovane u ćelije (`# %%`) i mogu se izvršavati ćeliju po ćeliju u VS Code-u, dok sveske sadrže sačuvane ispise i grafikone iz poslednjeg pokretanja.

## Faze

| Faza | Fajl | Opis |
|---|---|---|
| 0 | `00_redteam` | Istraživanje redteam evidencije, izdvajanje napadnutih naloga i raspodela napada kroz dane |
| 1 | `01_sample_and_label` | Formiranje uzorka iz `auth.txt` pomoću DuckDB-a i obeležavanje događaja na osnovu redteam evidencije |
| 2 | `02_features` | Normalizacija naloga na entitete, agregacija u jednosatne prozore, istorijski atributi i vremenska podela na skupove |
| 3 | `03_data_understanding_and_preparation` | Eksplorativna analiza, imputacija i selekcija atributa u tri koraka (53 → 40 → 11) |
| 4 | `04_classic_models` | Klasični modeli na tri nivoa skupa atributa, uticaj balansiranja klasa, podešavanje hiperparametara i izbor praga |
| 5 | `05_deep_models` | LSTM, GRU i Transformer nad sekvencama od 10 prozora, na sva tri skupa atributa |
| 6 | `06_unsupervised_models` | Isolation Forest i jednoklasna mašina sa potpornim vektorima |
| 7 | `07_test_evaluation` | Jednokratna evaluacija pobedničkih modela na test skupu, bootstrap intervali poverenja i normalizovan AP |
| 8 | `08_xai` | SHAP analiza najboljeg modela i mapiranje na MITRE ATT&CK |

## Pokretanje

```bash
git clone https://github.com/FilipMilovanovic/Lateral-Movement-Detection.git
cd Lateral-Movement-Detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Nakon što su podaci smešteni u `data/raw/`, faze se pokreću redom, jer svaka koristi međurezultate prethodne:

```bash
python src/00_redteam.py
python src/01_sample_and_label.py
python src/02_features.py
python src/03_data_understanding_and_preparation.py
python src/04_classic_models.py
python src/05_deep_models.py
python src/06_unsupervised_models.py
python src/07_test_evaluation.py
python src/08_xai.py
```

Duboki modeli se automatski treniraju na GPU-u ako je dostupan, a u suprotnom na procesoru. Seme slučajnih brojeva je fiksirano (`SEED = 7`), pa ponovno pokretanje daje iste rezultate.

## Metodološke napomene

- **Vremenska podela bez preklapanja:** trening (dani 1–12), validacija (dani 13–16), test (dani 17–59). Raniji period se koristi za obuku, srednji za izbor modela i praga, a najkasniji isključivo za završnu procenu, čime se sprečava curenje informacija iz budućnosti.
- **Test skup je korišćen jednom**, nakon što su sve odluke o modelima i pragovima već donete na validaciji.
- **Prirodna neravnoteža klasa** je zadržana u svim skupovima. Balansiranje je ispitano samo kroz težine klasa u funkciji greške, i to posebno za svaki model.
- **Prag odlučivanja** se bira maksimizacijom F-mere na validacionom skupu.

## Ograničenja

- Eksperiment je sproveden nad uzorkom od 460 entiteta, ne nad celim skupom podataka.
- Udeo napadnutih naloga u uzorku (~20%) je znatno veći nego u realnom okruženju, pa se apsolutne vrednosti preciznosti ne prenose na produkciju bez ponovne kalibracije praga.
- Modeli ne koriste topologiju mreže niti graf autentifikacionih veza.

## Referenca na skup podataka

Kent, A. D. (2015). *Comprehensive, Multi-Source Cyber-Security Events*. Los Alamos National Laboratory. https://doi.org/10.17021/1179829