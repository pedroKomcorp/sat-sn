import camelot
import pymupdf
import re
import streamlit as st
import pandas as pd
import tributos

class Parcela:
    """Represents a single installment/payment from a PGDAS extract."""
    def __init__(
        self,
        nome_arquivo: str,
        periodo: str,
        total: float,
        classificacao: str,
        valor: float
    ):
        self.nome_arquivo = nome_arquivo
        self.periodo = periodo
        self.total = total
        self.classificacao = classificacao
        self.valor = valor

def find_text_in_table(table_df, pattern):
    table_as_string = table_df.to_string()
    return re.search(pattern, table_as_string.replace('\n', ' '), re.IGNORECASE) is not None

@st.cache_data
def pgdas(uploaded_files):
    """
    Extracts data from uploaded PGDAS PDF files and returns the data and processing logs.
    """
    if not uploaded_files:
        return [], []

    all_parcels = []
    logs = [] # List to hold log messages

    for file in uploaded_files:
        file_name = file.name
        logs.append(f"🚀 **{file_name}**: Iniciando processamento...")
        # --- Periodo --- #
        periodo_str = 'Não Encontrado'
        try:
            file.seek(0)
            pdf_doc = pymupdf.open(stream=file.read(), filetype="pdf")
            text_as_string = pdf_doc[0].get_text()
            
            periodo_matches = re.search(r'Período de Apuração \(PA\):\s*(\d{2}/\d{4})', text_as_string)
            if periodo_matches:
                periodo_str = periodo_matches.group(1)
            else:
                periodo_matches = re.search(r'(\d{2}/\d{4})', text_as_string)
                if periodo_matches:
                    periodo_str = periodo_matches.group(0)
            
            if periodo_str == 'Não Encontrado':
                logs.append(f"⚠️ **{file_name}**: Não foi possível encontrar o 'Período de Apuração'.")
            else:
                logs.append(f"✔️ **{file_name}**: Período de apuração encontrado: {periodo_str}")
        except Exception as e:
            logs.append(f"❌ **{file_name}**: Falha Crítica ao ler o PDF para extrair o período. Detalhes: {e}")
            continue

        # --- Total e valor parcelas --- #
        try:
          tables = camelot.read_pdf(file, pages='all')
          try:
            receita_bruta_pa = tables[0].df
            total = receita_bruta_pa.iloc[1][3]
            try:
              for table in tables:
                table_df = table.df
                is_com_st = find_text_in_table(table_df, r"Com substituição tributária")
                is_sem_st = find_text_in_table(table_df, r"Sem substituição tributária")
                for idx, row in table_df.iterrows():
                    cell_value = str(row[0]).lower()
                    if 'parcela' in cell_value:
                        classification = "Indefinido"  # Default classification
                        valor = cell_value.split('r$ ')[1]
                        # --- 2. Determine the Classification ---
                        if is_sem_st:
                            classification = "Sem ST e Sem Monofasia"
                        elif is_com_st:
                            # For tables "Com substituição", look at the text *following* the parcela
                            classification_text_rows = []
                            next_row_index = idx + 1
                            
                            while next_row_index < len(table_df):
                                next_row_content = str(table_df.iloc[next_row_index][0]).lower()
                                if 'parcela' in next_row_content:
                                    break
                                classification_text_rows.append(next_row_content)
                                next_row_index += 1
                            
                            classification_text_joined = ' '.join(classification_text_rows)

                            # Check for keywords to assign a specific classification
                            has_icms = 'icms' in classification_text_joined
                            has_cofins = 'cofins' in classification_text_joined
                            has_pis = 'pis' in classification_text_joined

                            if has_icms and has_cofins and has_pis:
                                classification = "Com ST e Monofasia"
                            elif has_icms and not (has_cofins or has_pis):
                                classification = "Com ST (ICMS)"
                            elif not has_icms and (has_cofins and has_pis):
                                classification = "Com Monofasia (PIS/COFINS)"
                            else:
                                classification = "Com ST (Classificação Mista/Outra)"
                            
                        # --- 3. Create and append the Parcela object ---
                        parcela = Parcela(
                            nome_arquivo=file_name,
                            periodo=periodo_str,
                            total=total,
                            classificacao=classification, # Use the determined classification
                            valor=valor
                        )
                        all_parcels.append(parcela)
            except Exception as e:
              logs.append(f"❌ **{file_name}**: Erro na captação das parcelas. Procurando demais parcelas. Detalhes: {e}")
              continue
          except Exception as e:
            logs.append(f"❌ **{file_name}**: Falha Crítica ao extrair valor total do PDF. Arquivo será ignorado. Detalhes: {e}")
            continue
        except:
          logs.append(f"❌ **{file_name}**: Falha Crítica ao extrair tabelas do PDF. Arquivo será ignorado. Detalhes: {e}")
          continue
    return all_parcels, logs

@st.cache_data
def sieg(uploaded_files):
    if not uploaded_files:
        return pd.DataFrame()

    all_dfs = []

    # Process uploaded files
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_excel(
                uploaded_file,
                sheet_name="PRODUTOS",
                engine="calamine",
                dtype={"NCM": str}
            )

            required_cols = ["CFOP", "NCM", "DATA EMISSAO"]
            if not all(col in df.columns for col in required_cols):
                st.error(
                    f"Erro em '{uploaded_file.name}': "
                    f"A planilha 'PRODUTOS' deve conter as colunas: {', '.join(required_cols)}. "
                    "Este arquivo será ignorado."
                )
                continue

            # Padroniza colunas importantes
            df["NCM"] = df["NCM"].astype(str).str.strip()
            df["DATA EMISSAO"] = pd.to_datetime(df["DATA EMISSAO"], errors="coerce")

            all_dfs.append(df)

        except Exception as e:
            st.error(
                f"Não foi possível ler o arquivo '{uploaded_file.name}'. "
                f"Verifique o formato e se a planilha 'PRODUTOS' existe. Detalhes: {e}"
            )
            continue

    if not all_dfs:
        st.warning("Nenhum arquivo SIEG válido foi processado.")
        return pd.DataFrame()

    # Junta todos os arquivos
    final_df = pd.concat(all_dfs, ignore_index=True)


    # --- PIS/COFINS ---
    piscofins_df = tributos.piscofins()
    piscofins_df["NCM"] = piscofins_df["NCM"].astype(str).str.strip()
    piscofins_indexed = piscofins_df.set_index("NCM")["TRIBUTAÇÃO"]

    final_df["PIS/COFINS"] = final_df["NCM"].map(piscofins_indexed)
    final_df["PIS/COFINS"] = final_df["PIS/COFINS"].fillna("NCM NÃO IDENTIFICADO")


    # --- ICMS ---
    icms_df = tributos.icms()
    icms_df["NCM"] = icms_df["NCM"].astype(str).str.strip()
    icms_df["INICIO"] = pd.to_datetime(icms_df["INICIO"], dayfirst=True, errors="coerce")
    icms_df["FIM"] = pd.to_datetime(icms_df["FIM"], dayfirst=True, errors="coerce")

    merged = final_df.merge(
    icms_df,
    on="NCM",
    how="left",
    suffixes=("", "_icms")
    )

    # Filtra por intervalo de datas
    mask = (merged["DATA EMISSAO"] >= merged["INICIO"]) & (merged["DATA EMISSAO"] <= merged["FIM"])
    merged = merged[mask]

    # Garante uma única classificação por linha original
    icms_map = merged.groupby(merged.index)["CLASSIFICAÇÃO"].first()

    # Cria a nova coluna alinhada com df_sieg
    final_df["ICMS"] = final_df.index.map(icms_map)

    
    return final_df