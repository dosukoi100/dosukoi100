#Streamlitでの物体検出(Custom Vision)アプリ

#外部ライブラリーのインポート

#!pip3 install azure-cognitiveservices-vision-customvision

#ライブラリー、モジュール周りをインポート

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid ,json
import numpy as np

#jsonファイルを展開してトレーニングのエンドポイント、キー、予測のキーとリソースIDを変数に格納
with open('full_hel_arm.json') as f:
    secret = json.load(f)

ENDPOINT = secret['ENDPOINT']
TRAINING_KEY = secret['TRAINING_KEY']
ENDPOINT1 = secret['ENDPOINT1']
PREDICTION_KEY = secret['PREDICTION_KEY']
PREDICTION_RESOURCE_ID = secret['PREDICTION_RESOURCE_ID']

#クライアント認証をする 

credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY}) 

trainer = CustomVisionTrainingClient(ENDPOINT, credentials) 

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY}) 

predictor = CustomVisionPredictionClient(ENDPOINT1, prediction_credentials) 

#必要なコードを集めていく

#モデル名(イテレーションの名前)を明示
publish_iteration_name = "full-hel-arm"

#プロジェクトIDをproject_idとする

project_id = "プロジェクトIDを指定"

#画像をアップロードする→ファイルパスの指定→ファイルの保存
#lists=[]を作りタグ名、信頼度、座標情報取得→座標情報、タグ名を変数に代入

#ここからstreamlitのコードを書いて行きます
#streamlitのインポート
import streamlit as st

#タイトルの表示
st.title('物体検出アプリ(・ω・)/')
st.write('１回１兆円なりよ(・ω・)/')
#矩形を描くためのライブラリー
from PIL import ImageDraw
#フォント周りのライブラリー
from PIL import ImageFont
#画像を読み取るライブラリー
from PIL import Image

#ファイルのアップロード機能を作成
#streamlitのファイルをアップロードする関数file_upload()関数を
#使ってアップロード機能を実装します。第一引数にコメント、第二引数
#に許可するファイル形式を指定します。
uploaded_file = st.file_uploader('画像イメージを指定してください',type=['jpg','png','jpeg'])

if uploaded_file is not None:
    #PythonライブラリーPILからImageライブラリーを使用して
    #open関数を使用してファイルを開く
    img = Image.open(uploaded_file)
    #ファイル名だけのものにパスを用意する
    img_path = f'pod/{uploaded_file.name}'#←.nameでファイル名を取得
    #PILのImageの.save()関数を使用して画像を保存します。
    img.save(img_path)
    
    #numpyを入れる
    #ここは、インド人女性のyoutubeから参照
    #numpyのshapeで行、列、(chは?)の値を格納する(後学習のnumpより)
    
    image = Image.open(img_path)
    hig,wid,ch = np.array(image).shape
    
    
    #予測エンドポイントをテストする
    with open(img_path, mode="rb") as test_data:
        results = predictor.detect_image(project_id, publish_iteration_name, test_data)

    import datetime
    print(datetime.datetime.now())
    #ここら辺でタグに名前をつけて信頼度を付け、座標情報をとってきている
    #オリジナル
    lists =[]#空のの配列lists
    for prediction in results.predictions:
        if (prediction.probability * 100 > 10): #テストなら10本番なら約70位
            print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
            lists.append(prediction)#results.predictionsのタグ名、座標情報を格納
            print('-----------------------------')
            print (prediction.tag_name,'||',prediction.probability * 100)
    #for prediction in results.predictions:
    #    print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
    #    lists.append(prediction)
    
    
    
    #↑↑↑↑↑この配列listsの中に、タグの名前や信頼度、座標情報が入っている
    
    
    #描画(矩形を描く)
    #描画を描く為にImageDrawのDraw()関数を使います。
    draw = ImageDraw.Draw(img)
    for object in lists:
        x = object.bounding_box.left * wid
        y = object.bounding_box.top * hig
        w = object.bounding_box.width * wid
        h = object.bounding_box.height * hig
        print(x,'|',y,'|',w,'|',h)
        #配列objectsのobjectを取得する。ただし、現在はobject_property
        #caption = object.object
        tagname = object.tag_name
        
        #(順序2)検出した物質の名前(属性)を表示するためにもう１つ矩形を作り
        #その矩形の上に名前(属性)を表示する。
        #PILのImageFontからtruetype()関数を使用してttfファイルを読み込み
        #サイズを50とする
        font1 = ImageFont.truetype(font='./Helvetica 400.ttf',size=50)
        #検出した物質の名前(属性)のフォントサイズ(文字の大きさ(文字数))
        #ImageDrawのtextsize()関数を使用して抽出する。第一引数に抽出
        #する文字列、第二引数にフォントを指定する。フォントサイズは幅と
        #高さを返します。
        if ((object.probability * 100) >= 70):#最初prediction.probabilityとしてハマった
            text_w,text_h = draw.textsize(tagname,font=font1)
        #(順序1)矩形を描くためにImageDrawのrectangle()関数を使用する
        #第一引数にリスト・タプル型で始点、終点を指定、第二引数で塗り潰しの指定、
        #第三引数で矩形の色、第四引数で矩形の線の幅を指定する
            draw.rectangle([(x,y),(x+w,y+h)],fill=None,outline='green',width=5)
        #(順序2の続き)順序2の句形を作って行きます。塗りつぶしなのでoutline以降は不要です
            draw.rectangle([(x,y),(x+text_w,y+text_h)],fill='green')
        #矩形に文字を貼り付けます。第一引数に座標情報の始点、第二引数に文字列
        #第三引数にフォント情報(教材には無かったが追加)、第四引数に文字の色を指定します
            draw.text((x,y),tagname,font=font1,fill='white')
        elif ((object.probability * 100) < 70 and (prediction.probability * 100) >= 50):
            text_w,text_h = draw.textsize(tagname,font=font1)
            draw.rectangle([(x,y),(x+w,y+h)],fill=None,outline='yellow',width=5)
            draw.rectangle([(x,y),(x+text_w,y+text_h)],fill='yellow')
            draw.text((x,y),tagname,font=font1,fill='green')
        elif ((object.probability * 100) < 50):
            text_w,text_h = draw.textsize(tagname,font=font1)
            draw.rectangle([(x,y),(x+w,y+h)],fill=None,outline='red',width=5)
            draw.rectangle([(x,y),(x+text_w,y+text_h)],fill='red')
            draw.text((x,y),tagname,font=font1,fill='yellow')
            
    
    
    #draw.rectangle([(100,500),(1000,5000)],fill=None,outline='green',width=5)
    #streamlitのimage()関数を使用してimgを表示
    st.image(img)
    
    #タグの名前を表示する
    
    """
    
    ## 矩形は信頼度によって色が変わります
    ### 緑色      信頼度 70%以上
    ### 黄色      信頼度 70%未満50%以上
    ### 赤色      信頼度 50%未満
    
    """
    
    for i in lists:
        if (i.probability * 100 >= 70):
            st.write('{}の信頼度は{}です'.format(i.tag_name,(i.probability * 100)))



