"""
report_generator.py - Professional Multi-Format Report Export Engine
Generates downloadable Excel (.xlsx), CSV (.csv), and PDF (.pdf) reports for:
- Demand Forecast & Inventory Status
- Reorder Recommendations & Safety Stock
- Overstock & Working Capital Optimization
- Supplier Performance Audits
- Model Performance Tracking
"""
import io
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_csv_report(data_list, report_type='forecast'):
    """Generates standard CSV report."""
    df = pd.DataFrame(data_list)
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    bio.seek(0)
    filename = f"{report_type}_report_{datetime.date.today().strftime('%Y%m%d')}.csv"
    return bio, 'text/csv', filename

def generate_excel_report(data_list, report_type='forecast'):
    """Generates styled Excel report with openpyxl."""
    df = pd.DataFrame(data_list)
    bio = io.BytesIO()
    
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        sheet_name = report_type.replace('_', ' ').title()[:30]
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # Access openpyxl workbook & worksheet to style headers
        ws = writer.sheets[sheet_name]
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

        header_font = Font(name='Arial', size=11, bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='2A2A3D', end_color='2A2A3D', fill_type='solid')
        header_align = Alignment(horizontal='center', vertical='center', wrap_text=True)

        for col_num in range(1, len(df.columns) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
            # Adjust column width
            max_len = max(len(str(val)) for val in [cell.value] + list(df.iloc[:, col_num - 1]))
            col_letter = cell.column_letter
            ws.column_dimensions[col_letter].width = max(14, min(max_len + 4, 40))

    bio.seek(0)
    filename = f"{report_type}_report_{datetime.date.today().strftime('%Y%m%d')}.xlsx"
    return bio, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename

def generate_pdf_report(data_list, report_type='forecast', title='Inventory Demand & Optimization Report'):
    """Generates clean, professional PDF report using ReportLab."""
    bio = io.BytesIO()
    doc = SimpleDocTemplate(
        bio,
        pagesize=landscape(letter),
        rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a1a26')
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#555566')
    )
    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1 # Center
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#222222')
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph(f"DemandIQ™ — {title}", title_style))
    today_str = datetime.datetime.now().strftime('%B %d, %Y - %H:%M UTC')
    story.append(Paragraph(f"Generated on {today_str} | Confidential Enterprise Report", meta_style))
    story.append(Spacer(1, 16))

    if not data_list:
        story.append(Paragraph("No records available to report.", meta_style))
    else:
        # Select top columns to fit landscape table cleanly
        df = pd.DataFrame(data_list)
        all_cols = list(df.columns)
        
        # Limit columns to max 8 for landscape readability
        display_cols = all_cols[:8]
        
        table_data = []
        # Header row
        header_row = [Paragraph(str(c).replace('_', ' ').title(), table_header_style) for c in display_cols]
        table_data.append(header_row)

        for _, row in df.iterrows():
            row_cells = []
            for col in display_cols:
                val = row[col]
                if isinstance(val, float):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)
                row_cells.append(Paragraph(val_str, table_cell_style))
            table_data.append(row_cells)

        # Build table
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#222238')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f7f7fc')]),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t)

    doc.build(story)
    bio.seek(0)
    filename = f"{report_type}_report_{datetime.date.today().strftime('%Y%m%d')}.pdf"
    return bio, 'application/pdf', filename
