from ninja.files import UploadedFile
from django.http import HttpRequest
from ninja import Router, File, Form
from .schemas import NstResponse, NstRequest
from nstapp.services.nst_service import nst_apply, nst2_apply, nst3_apply, nst4_apply

router = Router(tags=["YouHi"])


@router.post("/", response=NstResponse)
def kandinsky(request: HttpRequest, nst_request: NstRequest = Form(...), img: UploadedFile = File(...)) -> dict:
    file_url = nst_apply(nst_request.key, img)
    return {"file_url": file_url}


@router.post("/mosaic", response=NstResponse)
def mosaic(request: HttpRequest, nst_request: NstRequest = Form(...), img: UploadedFile = File(...)) -> dict:
    file_url = nst2_apply(nst_request.key, img)
    return {"file_url": file_url}


@router.post("/piccasso", response=NstResponse)
def piccasso(request: HttpRequest, nst_request: NstRequest = Form(...), img: UploadedFile = File(...)) -> dict:
    file_url = nst3_apply(nst_request.key, img)
    return {"file_url": file_url}


@router.post("/monet", response=NstResponse)
def monet(request: HttpRequest, nst_request: NstRequest = Form(...), img: UploadedFile = File(...)) -> dict:
    file_url = nst4_apply(nst_request.key, img)
    return {"file_url": file_url}
