import time
import streamlit as st
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import extract as extract

# ------------------------------
# Utilidades
# ------------------------------

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Adicionar filtros")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrar tabela em:", df.columns, placeholder='Escolha as opções')
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valores para {column}",
                    df[column].unique(),
                    placeholder='Escolha as opções',
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100 if _max != _min else 1.0
                user_num_input = right.slider(
                    f"Valores para {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valores para {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


def _build_valor_nfe(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a coluna VALOR NFE de forma idempotente para o DF recebido."""
    if df.empty:
        df["VALOR NFE"] = 0
        return df

    df = df.copy()
    df["VALOR NFE"] = 0
    cols_to_add = [
        'VALOR TOTAL', 'VALOR FRETE', 'VALOR SEGURO',
        'DESPESAS ACESSORIAS', 'VALOR ST', 'VALOR FCPST',
        'VALOR IPI', 'IPI DEVOLVIDO', 'VLR II'
    ]
    cols_to_subtract = ['VALOR DESCONTO']

    for col in cols_to_add:
        if col in df.columns:
            df['VALOR NFE'] += df[col].fillna(0)
    for col in cols_to_subtract:
        if col in df.columns:
            df['VALOR NFE'] -= df[col].fillna(0)
    return df


# ------------------------------
# Callbacks reativos
# ------------------------------

def compute_faturamento():
    base = st.session_state.get("sieg_view", st.session_state.get("sieg_data", pd.DataFrame()))
    cfops = st.session_state.get("faturamento_selector", [])
    if base is None:
        base = pd.DataFrame()

    if cfops:
        df = base[base['CFOP'].isin(cfops)].copy()
        st.session_state.df_faturamento = _build_valor_nfe(df)
    else:
        st.session_state.df_faturamento = pd.DataFrame()

    # marcar uma versão para forçar dependências a recarregarem
    st.session_state["_apuracao_version"] = st.session_state.get("_apuracao_version", 0) + 1


def compute_devolucao():
    base = st.session_state.get("sieg_view", st.session_state.get("sieg_data", pd.DataFrame()))
    cfops = st.session_state.get("devolucao_selector", [])
    if base is None:
        base = pd.DataFrame()

    if cfops:
        df = base[base['CFOP'].isin(cfops)].copy()
        st.session_state.df_devolucao = _build_valor_nfe(df)
    else:
        st.session_state.df_devolucao = pd.DataFrame()

    st.session_state["_apuracao_version"] = st.session_state.get("_apuracao_version", 0) + 1


# ------------------------------
# App
# ------------------------------

def __init__():
    # Estado inicial
    if 'sieg_data' not in st.session_state:
        st.session_state.sieg_data = pd.DataFrame()
    if 'pgdas_data' not in st.session_state:
        st.session_state.pgdas_data = []
    if 'df_faturamento' not in st.session_state:
        st.session_state.df_faturamento = pd.DataFrame()
    if 'df_devolucao' not in st.session_state:
        st.session_state.df_devolucao = pd.DataFrame()
    if 'processing_reports' not in st.session_state:
        st.session_state.processing_reports = []
    if '_apuracao_version' not in st.session_state:
        st.session_state._apuracao_version = 0

    # --- Main App ---
    st.set_page_config(page_title='SAT-SN', layout="wide")
    st.title('📊 SAT-SN: Sistema de Apuração Tributária do Simples Nacional')
    st.markdown("---")

    st.header('Importar Documentos')
    with st.expander('PGDAS e Planilha SIEG', expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            pgdas_files_uploaded = st.file_uploader("Selecione os extratos (.pdf)", type="pdf", accept_multiple_files=True)
        with col2:
            sieg_files_uploaded = st.file_uploader("Selecione a planilha (.xlsx)", type="xlsx", accept_multiple_files=True)

        def get_unique_files(files):
            unique_files = {}
            if files:
                for file in files:
                    if file.name not in unique_files:
                        unique_files[file.name] = file
                removed_files = set([f.name for f in files]) - set(unique_files.keys())
                return list(unique_files.values()), list(removed_files)
            return [], []

        pgdas_files, removed_pgdas = get_unique_files(pgdas_files_uploaded)
        if removed_pgdas:
            st.warning(f"Arquivos duplicados removidos (PGDAS): {', '.join(removed_pgdas)}")

        sieg_files, removed_sieg = get_unique_files(sieg_files_uploaded)
        if removed_sieg:
            st.warning(f"Arquivos duplicados removidos (SIEG): {', '.join(removed_sieg)}")

        if st.button('Processar Arquivos'):
            # Reset state before processing new files
            st.session_state.sieg_data = pd.DataFrame()
            st.session_state.pgdas_data = []
            st.session_state.processing_reports = []
            st.session_state.df_faturamento = pd.DataFrame()
            st.session_state.df_devolucao = pd.DataFrame()
            st.session_state.sieg_view = pd.DataFrame()

            pgdas_data_all = []
            sieg_data_all = pd.DataFrame()
            
            total_files = len(pgdas_files) + len(sieg_files)
            progress_bar = st.progress(0)
            progress_status = st.empty()
            files_processed = 0

            # PGDAS Processing
            if pgdas_files:
                for file in pgdas_files:
                    files_processed += 1
                    progress_text = f"Processando arquivo {files_processed} de {total_files}: {file.name}"
                    progress_status.text(progress_text)
                    try:
                        dados_pgdas, pgdas_logs = extract.pgdas([file])
                        if dados_pgdas:
                            pgdas_data_all.extend(dados_pgdas)
                            st.session_state.processing_reports.append(f"✅ **PGDAS - {file.name}**: Processado com sucesso!")
                        else:
                            error_message = pgdas_logs[0].get('message', 'Erro desconhecido.') if pgdas_logs else 'Erro desconhecido.'
                            st.session_state.processing_reports.append(f"❌ **PGDAS - {file.name}**: Falha - {error_message}")
                    except Exception as e:
                        st.session_state.processing_reports.append(f"❌ **PGDAS - {file.name}**: Erro inesperado - {e}")
                    progress_bar.progress(files_processed / total_files)
                st.session_state.pgdas_data = pgdas_data_all

            # SIEG Processing
            if sieg_files:
                for file in sieg_files:
                    files_processed += 1
                    progress_text = f"Processando arquivo {files_processed} de {total_files}: {file.name}"
                    progress_status.text(progress_text)
                    try:
                        dados_sieg_single = extract.sieg([file])
                        if not dados_sieg_single.empty:
                            sieg_data_all = pd.concat([sieg_data_all, dados_sieg_single], ignore_index=True)
                            st.session_state.processing_reports.append(f"✅ **SIEG - {file.name}**: Processado com sucesso!")
                        else:
                            st.session_state.processing_reports.append(f"❌ **SIEG - {file.name}**: Falha no processamento ou arquivo vazio.")
                    except Exception as e:
                        st.session_state.processing_reports.append(f"❌ **SIEG - {file.name}**: Erro inesperado - {e}")
                    progress_bar.progress(files_processed / total_files)
                st.session_state.sieg_data = sieg_data_all

            if not pgdas_files and not sieg_files:
                st.warning("Nenhum arquivo selecionado para processamento.")
            else:
                progress_status.text("Processamento concluído!")
                time.sleep(2)
                progress_status.empty()
                progress_bar.empty()

    if st.session_state.processing_reports:
        with st.expander('Relatório de Processamento', expanded=False):
            # Make the report scrollable with a fixed height
            st.markdown(
            """
            <div style="max-height: 300px; overflow-y: auto; padding-right: 10px;">
            """,
            unsafe_allow_html=True
            )
            for report in st.session_state.processing_reports:
                if "✅" in report:
                    st.success(report)
                else:
                    st.error(report)
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("SIEG")
    if not st.session_state.sieg_data.empty:
        with st.expander('Valores Sieg'):
            sieg_view = filter_dataframe(st.session_state.sieg_data)
            st.session_state.sieg_view = sieg_view  # base para os cálculos
            st.dataframe(sieg_view)

        with st.expander('Faturamento e Devolução'):
            all_cfops = sorted(st.session_state.sieg_data['CFOP'].unique())
            col1, col2 = st.columns(2)

            # --- Faturamento ---
            with col1:
                st.subheader("Faturamento")
                st.multiselect(
                    label="Selecione os CFOPs de faturamento:",
                    options=all_cfops,
                    key='faturamento_selector',
                    on_change=compute_faturamento,
                )

                # exibir resultados atuais
                if not st.session_state.df_faturamento.empty:
                    pivot_table_faturamento = pd.pivot_table(
                        st.session_state.df_faturamento, index='CFOP', values=['VALOR NFE'], aggfunc='sum'
                    )
                    total_valor_nfe = st.session_state.df_faturamento['VALOR NFE'].sum()
                    with st.expander("Visualizar Tabela de Faturamento", expanded=True):
                        st.dataframe(pivot_table_faturamento.style.format("R$ {:,.2f}"), use_container_width=True)
                        st.metric(label="**TOTAL GERAL FATURAMENTO**", value=f"R$ {total_valor_nfe:,.2f}")
                else:
                    st.info("Selecione um ou mais CFOPs para ver a análise de faturamento.")

            # --- Devolução ---
            with col2:
                st.subheader("Devolução")
                st.multiselect(
                    label="Selecione os CFOPs de devolução:",
                    options=all_cfops,
                    key='devolucao_selector',
                    on_change=compute_devolucao,
                )

                if not st.session_state.df_devolucao.empty:
                    pivot_table_devolucao = pd.pivot_table(
                        st.session_state.df_devolucao, index='CFOP', values=['VALOR NFE'], aggfunc='sum'
                    )
                    total_valor_nfe_dev = st.session_state.df_devolucao['VALOR NFE'].sum()

                    with st.expander("Visualizar Tabela de Devolução", expanded=True):
                        st.dataframe(pivot_table_devolucao.style.format("R$ {:,.2f}"), use_container_width=True)
                        st.metric(label="**TOTAL GERAL DEVOLUÇÃO**", value=f"R$ {total_valor_nfe_dev:,.2f}")
                else:
                    st.info("Selecione um ou mais CFOPs para ver a análise de devolução.")

            # Se o usuário alterou os filtros (tabela de cima), recalcular com as seleções atuais
            if 'faturamento_selector' in st.session_state:
                compute_faturamento()
            if 'devolucao_selector' in st.session_state:
                compute_devolucao()

            st.markdown("<br>", unsafe_allow_html=True)
            if not st.session_state.df_faturamento.empty:
                with st.expander("Ver detalhes do Faturamento"):
                    st.dataframe(st.session_state.df_faturamento, use_container_width=True)
            if not st.session_state.df_devolucao.empty:
                with st.expander("Ver detalhes da Devolução"):
                    st.dataframe(st.session_state.df_devolucao, use_container_width=True)
    else:
        st.info('Importe os dados do SIEG')
    st.markdown("---")

    st.header("Apuração")
    # A simples leitura de '_apuracao_version' garante que a seção reaja às mudanças
    _ = st.session_state.get('_apuracao_version', 0)

    if st.session_state.pgdas_data or not st.session_state.sieg_data.empty:
        with st.expander('Resumo da Apuração', expanded=True):
            try:
                pgdas_list_of_dicts = [p.__dict__ for p in st.session_state.pgdas_data]
                df_pgdas = pd.DataFrame(pgdas_list_of_dicts)
                
                if 'total' in df_pgdas.columns:
                    df_pgdas['total'] = pd.to_numeric(
                        df_pgdas['total'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
                        errors='coerce'
                    )
                else:
                    st.error("Coluna 'total' não encontrada nos dados extraídos do PGDAS. Verifique a extração.")
                    st.stop()
                if 'periodo' in df_pgdas.columns:
                    df_pgdas['periodo'] = df_pgdas['periodo'].astype(str)
                    periodos = df_pgdas['periodo'].astype(str).unique()
                    anos = sorted({p[3:] for p in periodos})  # MM/YYYY

                    # --- Add Toggles for Filtering ---
                    col1, col2 = st.columns(2)
                    with col1:
                        show_correct = st.checkbox('Mostrar meses corretos', value=True)
                    with col2:
                        show_incorrect = st.checkbox('Mostrar meses com divergência', value=True)
                    
                    # Prepare SIEG data for matching
                    df_fat_copy = st.session_state.df_faturamento.copy()
                    df_dev_copy = st.session_state.df_devolucao.copy()
                    
                    if ('DATA EMISSAO' in df_fat_copy.columns) or ('DATA EMISSAO' in df_dev_copy.columns):
                        if 'DATA EMISSAO' in df_fat_copy.columns:
                            df_fat_copy['DATA EMISSAO'] = pd.to_datetime(df_fat_copy['DATA EMISSAO'], errors='coerce')
                        else:
                            df_fat_copy['DATA EMISSAO'] = pd.NaT  # placeholder column
                        
                        if 'DATA EMISSAO' in df_dev_copy.columns:
                            df_dev_copy['DATA EMISSAO'] = pd.to_datetime(df_dev_copy['DATA EMISSAO'], errors='coerce')
                        else:
                            df_dev_copy['DATA EMISSAO'] = pd.NaT  # placeholder column
                        # Distribute years evenly across columns, avoiding empty columns
                        num_cols = min(3, len(anos))
                        cols = st.columns(num_cols)
                        for idx, ano in enumerate(anos):
                            col = cols[idx % num_cols]
                            with col:
                                st.subheader(f"🗓️ {ano}")
                                with st.expander(''):
                                    periodos_do_ano = sorted({p for p in periodos if p.endswith(ano)})
                                    cols_p = st.columns(2)
                                    for idx_p, periodo in enumerate(periodos_do_ano):
                                        col = cols_p[idx_p % 2]
                                        with col:
                                            mes_str = periodo[:2]
                                            ano_str = periodo[3:]

                                            faturamento_mes = (
                                                df_fat_copy[
                                                    (df_fat_copy['DATA EMISSAO'].dt.month == int(mes_str)) &
                                                    (df_fat_copy['DATA EMISSAO'].dt.year == int(ano_str))
                                                ]['VALOR NFE'].sum()
                                                if 'VALOR NFE' in df_fat_copy.columns else 0
                                            )

                                            devolucao_mes = (
                                                df_dev_copy[
                                                    (df_dev_copy['DATA EMISSAO'].dt.month == int(mes_str)) &
                                                    (df_dev_copy['DATA EMISSAO'].dt.year == int(ano_str))
                                                ]['VALOR NFE'].sum()
                                                if 'VALOR NFE' in df_dev_copy.columns else 0
                                            )

                                            total_pgdas_series = df_pgdas[df_pgdas['periodo'] == periodo]['total']
                                            if not total_pgdas_series.empty:
                                                total_pgdas = total_pgdas_series.iloc[0]
                                                total_sieg = faturamento_mes - devolucao_mes

                                                is_correct = abs(total_pgdas - total_sieg) == 0
                                                if (is_correct and show_correct) or (not is_correct and show_incorrect):
                                                    border_color = "green" if is_correct else "red"
                                                    with st.expander(f'{periodo}', expanded=True):
                                                        st.markdown(f"""
                                                        <div style="border: 2px solid {border_color}; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                                                            <h4>Período: {periodo}</h4>
                                                            <p><strong>Total PGDAS:</strong> R$ {total_pgdas:,.2f}</p>
                                                            <p><strong>Faturamento SIEG:</strong> R$ {faturamento_mes:,.2f}</p>
                                                            <p><strong>Devolução SIEG:</strong> R$ {devolucao_mes:,.2f}</p>
                                                            <p><strong>Total SIEG (Fat - Dev):</strong> R$ {total_sieg:,.2f}</p>
                                                            <h5 style="color:{border_color};">Diferença: R$ {total_pgdas - total_sieg:,.2f}</h5>
                                                        </div>
                                                        """, unsafe_allow_html=True)

                    else:
                        st.warning("Filtre faturação ou devolução.")
                else:
                    st.warning("Coluna 'periodo' não encontrada nos dados do PGDAS.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao processar a apuração: {e}")
                st.warning("Verifique se os dados estão no formato esperado.")
    else:
        st.info('Importe os dados do PGDAS e do SIEG para a apuração.')
    st.markdown("---")


if __name__ == '__main__':
    __init__()
