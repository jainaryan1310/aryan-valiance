from datetime import date
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import firestore
from querying import generate_response, translate
import nltk

nltk.download('punkt_tab')

app = FastAPI()

db = firestore.Client(project="kavach-440208")
collection_name = "ongc"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


async def check_active(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="No authorization header")

    auth = auth_header.split()
    if len(auth) != 2:
        raise HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )

    token = auth[1]
    try:
        payload = jwt.decode(token, "flos seges humilis", algorithms=["HS256"])
        return str(payload["id"])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/")
async def read_root():
    return {"code": 200, "response": "The server is running"}


@app.get("/start_session/")
# start a new user session
async def start_session(user_id: Annotated[str, Depends(check_active)]):
    return {"user_id": user_id, "response": "A Chat session has started"}


@app.get("/new_chat/")
# create a new chat document in the chat database
async def new_chat(user_id: Annotated[str, Depends(check_active)]):
    query = db.collection(collection_name).where("user_id", "==", user_id)
    docs = query.get()

    ucid = len(docs) + 1
    doc_id = f"{user_id}_{ucid}"

    chat_log = {
        "user_id": user_id,
        "ucid": ucid,
        "title": "new chat",
        "date": date.today().strftime("%d/%m/%y"),
        "message_history": [],
        "deleted": False,
    }

    doc_ref = db.collection(collection_name).document(doc_id)
    doc_ref.set(chat_log)

    response = JSONResponse(
        status_code=200,
        content={
            "user_id": user_id,
            "ucid": ucid,
            "response": "A new chat has been initialised",
        },
    )
    return response


@app.get("/edit_title/")
# edit the display title of a chat in the history
async def edit_title(
    user_id: Annotated[str, Depends(check_active)],
    ucid: Annotated[int, Query()],
    title: Annotated[str, Query()],
):
    doc_id = f"{user_id}_{ucid}"
    chat_doc = db.collection(collection_name).document(doc_id)

    try:
        chat_doc.set({"title": title}, merge=True)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update chat title\nError: {e}"
        )

    return {
        "user_id": user_id,
        "ucid": ucid,
        "title": title,
        "response": "The chat title has been updated",
    }


@app.get("/get_chats/")
# get the list of all chat by this user
async def get_chats(
    user_id: Annotated[str, Depends(check_active)],
):
    query = db.collection(collection_name).where("user_id", "==", user_id)
    docs = query.get()

    doc_list = []
    for doc in docs:
        d = doc.to_dict()
        if d["message_history"] != [] and d["deleted"] is False:
            # return chat_title instead of first message | also return date in response
            doc_list.append({"ucid": d["ucid"], "date": d["date"], "title": d["title"]})

    response = {"user_id": user_id, "doc_list": doc_list}
    return response


@app.get("/restore_chat/")
# restore and continue an old chat
async def restore_chat(
    user_id: Annotated[str, Depends(check_active)],
    ucid: Annotated[int, Query()],
):
    doc_id = f"{user_id}_{ucid}"
    chat_doc = db.collection(collection_name).document(doc_id)

    chat_log = chat_doc.get().to_dict()
    return chat_log


@app.get("/get_response/")
# get response for a new message
async def get_response(
    user_id: Annotated[str, Depends(check_active)],
    ucid: Annotated[int, Query()],
    user_input: Annotated[str, Query()],
    language: Annotated[str, Query()],
    api: Annotated[str, Query()],
    service: Annotated[str, Query()],
):
    doc_id = f"{user_id}_{ucid}"
    chat_doc = db.collection(collection_name).document(doc_id)
    chat_history = chat_doc.get().to_dict()["message_history"]

    if chat_history == []:
        chat_doc.set({"title": user_input}, merge=True)

    if language != "en":
        response_json = translate(user_input, language, "en")

        if response_json["code"] != 200:
            print(response_json)
            return JSONResponse(status_code=500, content=response_json)

        original_query = response_json["response"]

    else:
        original_query = user_input

    chat_history.append(
        {
            "author": "user",
            "content": user_input,
            "source": "None",
            "translated_content": original_query,
        }
    )

    response_json = generate_response(original_query, chat_history)

    if response_json["code"] != 200:
        print(response_json)
        return JSONResponse(status_code=500, content=response_json)

    if language != "en":
        translated_response = translate(response_json["text"], "en", language)

        if translated_response["code"] != 200:
            print(translated_response)
            return JSONResponse(status_code=500, content=response_json)

        translated_text = translated_response["response"]
    else:
        translated_text = response_json["text"]

    chat_history.append(
        {
            "author": "bot",
            "content": translated_text,
            "source": response_json["sources"],
            "images": response_json["images"],
            "translated_content": response_json["text"],
        }
    )

    message_history = {"message_history": chat_history}
    chat_doc.set(message_history, merge=True)

    return message_history
