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
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_excel(uploaded_file, sheet_name='PRODUTOS', engine='openpyxl', dtype={'NCM': str})
            required_cols = ['CFOP', 'NCM', 'DATA EMISSAO']
            
            if not all(col in df.columns for col in required_cols):
                st.error(
                    f"Erro em '{uploaded_file.name}': A planilha 'PRODUTOS' deve conter as colunas: "
                    f"{', '.join(required_cols)}. Este arquivo será ignorado."
                )
                continue
            all_dfs.append(df)

        except Exception as e:
            st.error(
                f"Não foi possível ler o arquivo '{uploaded_file.name}'. Verifique o formato "
                f"e se a planilha 'PRODUTOS' existe. Detalhes: {e}"
            )
            continue 
            
    if not all_dfs:
        st.warning("Nenhum arquivo SIEG válido foi processado.")
        return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)

    # --- PIS/COFINS Calculation ---
    piscofins_df = tributos.piscofins()
    
    final_df['NCM'] = final_df['NCM'].astype(str).str.strip()
    piscofins_df['NCM'] = piscofins_df['NCM'].astype(str).str.strip()

    piscofins_indexed = piscofins_df.set_index('NCM')['TRIBUTAÇÃO']

    # 2. Mapeie os valores para a coluna 'PIS COFINS'
    final_df['PIS/COFINS'] = final_df['NCM'].map(piscofins_indexed)
    final_df['PIS/COFINS'] = final_df['PIS/COFINS'].fillna('NCM NÃO IDENTIFICADO')

    final_df = final_df[['PIS/COFINS'] + [col for col in final_df.columns if col != 'PIS/COFINS']]

    # --- ICMS ---
    icms_df = tributos.icms()

    # Garantir NCM como string e sem espaços
    icms_df['NCM'] = icms_df['NCM'].astype(str).str.strip()

    # Renomear colunas apenas se necessário
    icms_df = icms_df.rename(columns={
        'Descricao': 'DESCRIÇÃO',
        'descricao': 'DESCRIÇÃO',
        'Cest': 'CEST',
        'cest': 'CEST',
        'Classificacao': 'CLASSIFICAÇÃO',
        'classificacao': 'CLASSIFICAÇÃO'
    })

    # Garantir que colunas obrigatórias existam
    for col in ['INICIO', 'FIM', 'CEST', 'DESCRIÇÃO', 'CLASSIFICAÇÃO']:
        if col not in icms_df.columns:
            icms_df[col] = None

    # Conversão de datas
    final_df['DATA EMISSAO'] = pd.to_datetime(final_df['DATA EMISSAO'], errors='coerce')
    icms_df['INICIO'] = pd.to_datetime(icms_df['INICIO'], errors='coerce').fillna(pd.to_datetime('1990-01-01'))
    icms_df['FIM'] = pd.to_datetime(icms_df['FIM'], errors='coerce').fillna(pd.to_datetime('2050-12-31'))

    # Merge
    merged_df = pd.merge(final_df, icms_df, on='NCM', how='left')

    # Função de classificação ICMS
    def classify_icms(row):
        if pd.isna(row["INICIO"]):
            return "NÃO ST"
        if row["INICIO"] <= row["DATA EMISSAO"] <= row["FIM"]:
            return row["CLASSIFICAÇÃO"] if pd.notna(row["CLASSIFICAÇÃO"]) else "NÃO ST"
        return "NÃO ST"

    # Criar coluna ICMS
    merged_df["ICMS"] = merged_df.apply(classify_icms, axis=1)

    # Reordenar para ICMS aparecer antes
    cols_to_move = ['ICMS']
    final_df = merged_df[cols_to_move + [col for col in merged_df.columns if col not in cols_to_move]]
    return final_df