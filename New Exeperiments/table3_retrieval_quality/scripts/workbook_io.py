"""Small dependency-free OOXML writer/reader for the annotation workbook."""
from pathlib import PurePosixPath
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
import re

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
def column(n):
    value=''
    while n:n,r=divmod(n-1,26);value=chr(65+r)+value
    return value

def write_workbook(path,headers,rows):
    last_column=column(len(headers))
    relevance_column=column(headers.index('relevance')+1)
    def cell(ref,value,style=0):
        if value is None or value=='':return f'<c r="{ref}" s="{style}"/>'
        text=str(value)
        if len(text)>32767:raise ValueError('Excel cell limit exceeded; source text must not be truncated')
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',text):raise ValueError('Invalid XML control character')
        return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'
    sheet=[f'<worksheet xmlns="{NS}"><dimension ref="A1:{last_column}{len(rows)+1}"/><sheetViews><sheetView workbookViewId="0"><pane xSplit="1" ySplit="1" topLeftCell="B2" activePane="bottomRight" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/><cols>']
    widths=dict(query_id=23,dataset=23,predicted_class=20,structured_query=75,retrieval_method=20,rank=9,chunk_id=45,document_id=35,chunk_text=95,relevance=14,occurrence_count=20)
    for i,h in enumerate(headers,1):sheet.append(f'<col min="{i}" max="{i}" width="{widths.get(h,20)}" customWidth="1"/>')
    sheet.append('</cols><sheetData><row r="1" ht="32" customHeight="1">'+''.join(cell(column(i)+'1',h,1) for i,h in enumerate(headers,1))+'</row>')
    for n,row in enumerate(rows,2):sheet.append(f'<row r="{n}" ht="120" customHeight="1">'+''.join(cell(column(i)+str(n),row[h],3 if h=='relevance' else 2) for i,h in enumerate(headers,1))+'</row>')
    sheet.append(f'</sheetData><autoFilter ref="A1:{last_column}{len(rows)+1}"/><dataValidations count="1"><dataValidation type="list" errorStyle="stop" allowBlank="1" showInputMessage="1" showErrorMessage="1" errorTitle="Binary label required" error="Enter 0 or 1 only." promptTitle="Human relevance judgment" prompt="Read the query and chunk. Enter 1 relevant or 0 not relevant." sqref="{relevance_column}2:{relevance_column}{len(rows)+1}"><formula1>"0,1"</formula1></dataValidation></dataValidations></worksheet>')
    files={
      '[Content_Types].xml':'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>',
      '_rels/.rels':'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
      'xl/workbook.xml':f'<workbook xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><bookViews><workbookView/></bookViews><sheets><sheet name="Annotations" sheetId="1" r:id="rId1"/></sheets></workbook>',
      'xl/_rels/workbook.xml.rels':'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
      'xl/styles.xml':f'<styleSheet xmlns="{NS}"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts><fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF243746"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="4"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" horizontal="center"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
      'xl/worksheets/sheet1.xml':''.join(sheet)}
    with ZipFile(path,'w',ZIP_DEFLATED) as archive:
        for name,value in files.items():archive.writestr(name,'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+value)

def read_workbook(path):
    with ZipFile(path) as z:
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        sheets=workbook.findall(f'{{{NS}}}sheets/{{{NS}}}sheet')
        if len(sheets)!=1 or sheets[0].attrib['name']!='Annotations':raise ValueError('Expected exactly one sheet named Annotations')
        rel=sheets[0].attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        relationships=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        item=next(r for r in relationships if r.attrib['Id']==rel)
        if item.attrib.get('TargetMode')=='External':raise ValueError('External sheet not allowed')
        target=item.attrib['Target'];target=target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl')/target)
        if '..' in PurePosixPath(target).parts:raise ValueError('Unsafe sheet path')
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        root=ET.fromstring(z.read(target));matrix=[]
        for row in root.findall(f'{{{NS}}}sheetData/{{{NS}}}row'):
            cells={}
            for c in row.findall(f'{{{NS}}}c'):
                if c.find(f'{{{NS}}}f') is not None:raise ValueError('Formula cells are not accepted as human annotations')
                index=0
                for letter in re.match(r'[A-Z]+',c.attrib['r']).group():index=index*26+ord(letter)-64
                kind=c.attrib.get('t');v=c.findtext(f'{{{NS}}}v',default='')
                if kind=='inlineStr':v=''.join(t.text or '' for t in c.findall(f'{{{NS}}}is//{{{NS}}}t'))
                elif kind=='s':v=strings[int(v)]
                elif kind=='e':raise ValueError('Excel error cell')
                elif kind=='b':raise ValueError('Boolean cell; enter numeric 0 or 1')
                elif kind in (None,'n') and v in ('0.0','1.0'):v=v[0]
                cells[index]=v
            if any(v!='' for v in cells.values()):matrix.append(cells)
        if not matrix:raise ValueError('Empty workbook')
        headers=[matrix[0].get(i,'') for i in range(1,max(matrix[0])+1)]
        if len(headers)!=len(set(headers)) or any(not h for h in headers):raise ValueError('Invalid workbook headers')
        result=[]
        for cells in matrix[1:]:
            if any(i>len(headers) and v!='' for i,v in cells.items()):raise ValueError('Unexpected extra cells')
            result.append({h:cells.get(i,'') for i,h in enumerate(headers,1)})
        return headers,result
