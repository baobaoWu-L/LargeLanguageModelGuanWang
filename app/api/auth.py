from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Header, status

from app.auth.auth import get_current_user
from app.db.auth import (get_user_by_username,create_user,update_last_login)
from app.common.auth import RegisterReq, LoginReq, TokenResp, UserInDB
# from app.db.mysql import get_conn
from app.secu.security import verify_password, create_access_token, decode_token


router = APIRouter(prefix="/auth", tags=["auth"])
@router.post("/register", response_model=UserInDB)
def register(req: RegisterReq):
    if get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="username already exists")
    u = create_user(req)
    return UserInDB(
        id=int(u["id"]),
        username=u["username"],
        email=u.get("email"),
        phone=u.get("phone"),
        full_name=u.get("full_name"),
        is_active=bool(u["is_active"]),
        is_super_admin=bool(u["is_super_admin"]),
    )

@router.post("/login", response_model=TokenResp)
def login(req: LoginReq):
    u = get_user_by_username(req.username)
    if not u:
        raise HTTPException(status_code=401, detail="bad credentials")

    if not verify_password(req.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="bad credentials")

    if not u.get("is_active"):
        raise HTTPException(status_code=403, detail="user disabled")

    update_last_login(int(u["id"]))
    token = create_access_token({"sub": u["username"], "uid": int(u["id"])})
    return TokenResp(access_token=token)

@router.get("/me", response_model=UserInDB)
def me(current_user: UserInDB = Depends(get_current_user)):
    return current_user