# from lxml import etree
# def parse_xml_to_dict(xml):
#
#     if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
#         return {xml.tag: xml.text}
#
#     result = {}
#     for child in xml:
#         child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
#         if child.tag != 'object':
#             result[child.tag] = child_result[child.tag]
#         else:
#             if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
#                 result[child.tag] = []
#             result[child.tag].append(child_result[child.tag])
#     return {xml.tag: result}
#
#
# xml_path=r"./VOCdevkit/VOC2012/Annotations/2007_000027.xml"
# with open(xml_path) as fid:
#     xml_str=fid.read()
# xml = etree.fromstring(xml_str)
# data=parse_xml_to_dict(xml)["annotation"]
# print(data)+
from PIL import Image
img_path='D:\\A_Lab_of_mine\\CV\\FasterRCNN_of_mine\\VOCdevkit\\VOC2012\\JPEGImages\\2008_000008.jpg'
image = Image.open(img_path)