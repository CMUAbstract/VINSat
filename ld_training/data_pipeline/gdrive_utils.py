import argparse
import logging
import sys
import time
from multiprocessing import Pool

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


def parse_args():
    """
    Parse command line arguments for labeling Data.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Downloading Data from GDrive")
    parser.add_argument(
        "--credentials-path",
        type=str,
        required=True,
        help="Path to credential json file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save downloaded files",
    )
    parser.add_argument(
        "--folders-to-download",
        nargs='+',
        default=[],
        required=False,
        help="Folder names to dowwnload, if not provided all folders will be downloaaded",
    )
    parsed_args = parser.parse_args()
    return parsed_args


def get_folder_list(drive_service):
    try:
        folders = []
        page_token = None
        while True:
            # pylint: disable=maybe-no-member
            response = (
                drive_service.files()
                .list(
                    q="mimeType='application/vnd.google-apps.folder'",
                    spaces="drive",
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )
            logging.debug(response)
            folders.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
    except HttpError as error:
        logging.error(f"An error occurred: {error}")
        folders = None
    return folders


def get_tiff_file_list(drive_service, folder_id):
    """Search file in drive location

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """

    query_string = "mimeType='image/tiff'"
    query_string += " and '" + folder_id + "' in parents"

    logging.debug(query_string)

    try:
        files = []
        page_token = None
        while True:
            # pylint: disable=maybe-no-member
            response = (
                drive_service.files()
                .list(
                    q=query_string,
                    spaces="drive",
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )
            logging.debug(response)
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
    except HttpError as error:
        logging.error(f"An error occurred: {error}")
        files = None
    return files


def download_file_from_google_drive(drive_service, file_id, destination):
    start = time.perf_counter()
    request = drive_service.files().get_media(fileId=file_id)
    with open(destination, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logging.debug("Download %d%%." % int(status.progress() * 100))
    download_time_seconds = time.perf_counter() - start
    logging.info(
        f"file id: {file_id} destination: {destination} download_time_seconds {download_time_seconds}"
    )


def multi_process_download(files):
    result = []
    pool = Pool(processes=200)
    res = pool.starmap(download_file_from_google_drive, files, chunksize=10)
    result.extend(res)
    pool.close()
    pool.join()
    logging.info(result)


if __name__ == "__main__":
    args = parse_args()
    logging.info(f"Args provided: {args}")
    creds = service_account.Credentials.from_service_account_file(
        args.credentials_path,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    drive_service = build("drive", "v3", credentials=creds)
    folders = get_folder_list(drive_service)
    logging.info(folders)
    filtered_folders = folders
    if len(args.folders_to_download) > 0:
        filtered_folders = [f for f in folders if f["name"] in args.folders_to_download]
    for folder in filtered_folders:
        files = get_tiff_file_list(drive_service, folder["id"])
        destination_dir = args.output_dir
        if files:
            mp_list = []
            for file in files:
                mp_list.append(
                    (drive_service, file["id"], f"{destination_dir}/{file['name']}")
                )
            logging.info(
                f"Number of files in folder {folder['name']}: {len(mp_list)}"
            )
            multi_process_download(mp_list)
