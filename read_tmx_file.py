from translate.storage.tmx import tmxfile

with open("/home/tomer/PycharmProjects/nlp_ex4/de_eg_ds.txt", 'rb') as fin:
    tmx_file = tmxfile(fin, 'de-DE', 'en-GB')

for node in tmx_file.unit_iter():
    print(node.source, node.target)

#
# import tmx
# file = tmx.TileMap.load('/home/tomer/PycharmProjects/nlp_ex4/de_eg_ds.txt')
# print(file)
# for val in file:
#   print(val)
# # with open('/home/tomer/PycharmProjects/nlp_ex4/de_eg_ds.txt', 'r') as file :
# #   filedata = file.read()
# #
# # # Replace the target string
# # filedata = filedata.replace('&#x14', ' ')
# #
# # # Write the file out again
# # with open('de_eg_ds.txt', 'w') as file:
# #   file.write(filedata)
