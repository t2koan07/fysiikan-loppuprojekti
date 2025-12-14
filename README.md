# Fysiikan loppuprojekti

Projektissa analysoidaan Phyphox-sovelluksella mitattua puhelimen kiihtyvyys- ja GPS-dataa.
Mittauksessa käytettiin Linear Acceleration- ja Location-sensoreita.

Analyysi on toteutettu Python-skriptinä Streamlit-kirjastolla.

## Menetelmä
- Kiihtyvyysdata suodatettiin kaistanpäästösuodattimella (noin 0.7–4 Hz)
- Askelmäärä määritettiin kahdella tavalla:
  1) Suodatetusta signaalista piikkien avulla
  2) Fourier-analyysin avulla tehospektrin päätaajuudesta
- Matka ja keskinopeus laskettiin GPS-datasta
- Askelpituus laskettiin jakamalla kuljettu matka askelmäärällä

## Visualisoinnit
- Suodatettu kiihtyvyysdata ja tunnistetut askeleet
- Tehospektritiheys (PSD)
- GPS-reitti kartalla

## Data
Repo sisältää datatiedostot:
- data/Accelerometer.csv
- data/Location.csv

Sovellus lukee datan ensisijaisesti paikallisesta data/-kansiosta.
Jos sovellus ajetaan suoraan GitHub-URL:stä, data haetaan raw.githubusercontent.com-osoitteesta.

## Käynnistys (paikallinen)
Asenna riippuvuudet:
python3 -m pip install -r requirements.txt

Käynnistä:
streamlit run app.py

## Käynnistys (URL-ajo)
Aja terminaalissa:
```bash
streamlit run https://raw.githubusercontent.com/t2koan07/fysiikan-loppuprojekti/main/app.py
