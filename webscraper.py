from bs4 import *
import requests as rq
import os

taj_mahal=rq.get("https://www.google.com/search?q=taj+mahal+images&rlz=1C1RLNS_enIN805IN805&source=lnms&tbm=isch&sa=X&ved=2ahUKEwitxbn4_93sAhVYmHIEHRMiD4cQ_AUoAXoECBYQAw&biw=1536&bih=754#imgrc=WOAQOY8RR9SBHM")
qutub_minar=rq.get("https://www.google.com/search?q=qutub+minar+images&tbm=isch&ved=2ahUKEwiG17D6_93sAhXdg3IEHW2YAEIQ2-cCegQIABAA&oq=qutub+images&gs_lcp=CgNpbWcQARgAMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB46BAgAEEM6AggAUMquYljbtWJgt71iaABwAHgAgAFRiAHkApIBATWYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=zeicX8a8KN2HytMP7bCCkAQ&bih=754&biw=1536&rlz=1C1RLNS_enIN805IN805")
india_gate=rq.get("https://www.google.com/search?q=india+gate+images&tbm=isch&ved=2ahUKEwjy2KH8hd7sAhV9qnIEHWBiDRAQ2-cCegQIABAA&oq=india+gate+images&gs_lcp=CgNpbWcQAzICCAAyAggAMgIIADICCAAyAggAMgIIADICCAAyBggAEAcQHjIGCAAQBxAeMgYIABAHEB46BAgAEEM6BggAEAoQGFDgzgFYufEBYOP1AWgEcAB4AIABcYgB_AeSAQQxMi4ymAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=HO-cX_LaDf3UytMP4MS1gAE&bih=754&biw=1536&rlz=1C1RLNS_enIN805IN805")
soup1=BeautifulSoup(taj_mahal.text,"html.parser")
soup2=BeautifulSoup(qutub_minar.text,"html.parser")
soup3=BeautifulSoup(india_gate.text,"html.parser")
x1=soup1.select('img[src^="https://"]')
x2=soup2.select('img[src^="https://"]')
x3=soup3.select('img[src^="https://"]')

links1,links2,links3=[],[],[]

for i in x1:
    links1.append(i['src'])
os.mkdir('tajmahal')

val1=1

for ind,link in enumerate(links1):
    if val1 <=20:
        data=rq.get(link).content
        with open('tajmahal/'+str(ind+1)+'.jpg','wb+') as f:
            f.write(data)
        val1+=1
    else:
        f.close()
        break

for j in x2:
    links2.append(j['src'])
os.mkdir('qutubminar')

val2=1

for ind,link in enumerate(links2):
    if val2 <=20:
        data=rq.get(link).content
        with open('qutubminar/'+str(ind+1)+'.jpg','wb+') as f:
            f.write(data)
        val2+=1
    else:
        f.close()
        break

for k in x3:
    links3.append(k['src'])
os.mkdir('indiagate')

val3=1

for ind,link in enumerate(links3):
    if val3 <=20:
        data=rq.get(link).content
        with open('indiagate/'+str(ind+1)+'.jpg','wb+') as f:
            f.write(data)
        val3+=1
    else:
        f.close()
        break
