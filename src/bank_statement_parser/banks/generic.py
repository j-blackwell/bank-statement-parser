from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd
import polars as pl
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption


class BankParserGeneric(object):
    def __init__(self) -> None:
        pipeline_options = PdfPipelineOptions(do_table_structure=True, generate_page_images=True)
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def convert_doc(self, source: Path) -> ConversionResult:
        result = self.doc_converter.convert(source=source)
        return result


    def get_tables(self, result: ConversionResult) -> List[pd.DataFrame]:
        tables = [table.export_to_dataframe() for table in result.document.tables]
        return tables

    @abstractmethod
    def tables_to_statement(self, tables: List[pd.DataFrame], **kwargs) -> Optional[pl.DataFrame]:
        pass

    def get_statement(self, source: Path, **kwargs) -> Optional[pl.DataFrame]:
        result = self.convert_doc(source)
        tables = self.get_tables(result)
        statement = self.tables_to_statement(tables, **kwargs)
        return statement

