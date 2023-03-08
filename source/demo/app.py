import os, sys
from libs import *

from api import NER

ner = NER("/home/ubuntu/khiem.lh/Free/ViNER/ckps/AccountingNER/word/")
examples = [
    'Ngày 20/10, nhập kho vật liệu F có trị giá 550.000.000 đồng (đã có thuế GTGT 10%), đã thanh toán 1/2 số tiền cho P.', 
    'Quản lý hoá đơn điện tử của các nhà cung cấp khác', 
    'Chỉ tiêu: Số tiền đã thanh toán bằng tiền gửi ngân hàng: Năm 2015: 1.250.000.000đ', 
    'Ngày 28/6, mua vật liệu M với số lượng mua 6.000 mét, đơn giá 24.000 đồng/mét, thuế GTGT 10%, đã nhận được hoá đơn nhưng cuối tháng số hàng này chưa về nhập kho.', 
    'Đối với phiếu xuất kho theo Nghị định 51/2010/NĐ-CP', 
    'Có TK 244 Cầm cố, thế chấp, kỹ quỹ, ký cược (TT 200)', 
    'Chi tiết chi phí: Chi phí khác bằng tiền mặt (chưa gồm 10% thuế GTGT); Phân xưởng 1: 86.000.000 đồng; Phân xưởng 2: 92.000.000 đồng.', 
    'Mẫu hoá đơn: cho phép thực hiện chức năng Lấy dữ liệu mẫu hoá đơn từ MISA meInvoice.', 
    'Hoá đơn đã thanh toán bằng tiền mặt cho bên vận chuyển 2.750.000 đồng (bao gồm thuế GTGT 10%).', 
    'Định kỳ, đơn vị cần lập bảng tổng hợp dự liệu hoá đơn điện tử gửi tới cơ quan thuế.', 
]

demo = gr.Interface(
    title = "Accounting Named Entity Recognition For Vietnamese", 
    description = "This is an easy-to-use interface built in Gradio for demonstrating a NER system that identifies Accounting related named entities in Vietnamese", 
    inputs = gr.Textbox(label = "Text", placeholder = "Paste your text here..."), outputs = gr.JSON(label = "Entities"), 
    fn = ner.ner_predict, 
    examples = examples, 
)
demo.launch()