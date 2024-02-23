from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import aiohttp
import asyncio
import time

app = FastAPI()

RUNPOD_URL = "http://localhost:8000"

async def stream_generator(headers, check_url):
    results = []
    async with aiohttp.ClientSession() as session:
        while True:       
            async with session.post(check_url, headers=headers) as response:
                result = await response.json()
                for chunk in result.get('stream', []):
                    yield chunk["output"].encode()
                    results.append(chunk["output"].encode())
                if result.get('status') == 'COMPLETED':
                    print(results)
                    break
            await asyncio.sleep(0.01)


def convert_input(request_data, route):
    converted = {"input": {"openai_route": route, "openai_input": request_data}}
    return converted


async def fetch_response( request_data, headers, route, stream=False):
    # url = f'https://api.runpod.ai/v2/{endpoint_id}'
    url = RUNPOD_URL
    post_url = f'{url}/run'
    async with aiohttp.ClientSession() as session:
        async with session.post(post_url, headers=headers, json=convert_input(request_data, route)) as post_response:
            if post_response.status != 200:
                raise HTTPException(status_code=post_response.status, detail="Failed to initialize the request")
            job_id = (await post_response.json()).get("id")
            check_url = f'{url}/{"stream" if stream else "status"}/{job_id}'

    if stream:
        # Return the async generator directly without passing the session
        return stream_generator(headers, check_url)
    else:
        # For non-streaming, you need to ensure session usage is also managed correctly
        
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.post(check_url, headers=headers) as response:
                    result = await response.json()
                    if result.get('status') == 'COMPLETED':
                        # print(f"Non-Stream Time: {time.time() - start}")
                        
                        # print(f"Total Time: {time.time() - start_time}")
                        return result.get('output')[0]
                await asyncio.sleep(0.01)

@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    
    start_time = time.time()
    headers = {'Authorization': request.headers.get('Authorization'), 'Content-Type': 'application/json'}
    if not headers['Authorization']:
        raise HTTPException(status_code=400, detail="Authorization header is missing")
    
    request_data = await request.json()
    print(request_data)
    stream = request_data.get("stream", False)

    if stream:
        generator = await fetch_response(request_data, headers, "/v1/chat/completions", stream=True)
        return StreamingResponse(generator)  # Make sure generator is called as a function
    else:
        result = await fetch_response(request_data, headers, "/v1/chat/completions")
        return JSONResponse(result)  # This now correctly waits for and passes the result



@app.post("/v1/completions")
async def completion(request: Request):
    start_time = time.time()
    headers = {'Authorization': request.headers.get('Authorization'), 'Content-Type': 'application/json'}
    if not headers['Authorization']:
        raise HTTPException(status_code=400, detail="Authorization header is missing")
    
    request_data = await request.json()
    print(request_data)
    stream = request_data.get("stream", False)

    if stream:
        generator = await fetch_response(request_data, headers, "/v1/chat/completions", stream=True)
        return StreamingResponse(generator)  # Make sure generator is called as a function
    else:
        result = await fetch_response(request_data, headers, "/v1/chat/completions")
        return JSONResponse(result)  # This now correctly waits for and passes the result

# @app.get("/health")
# async def get_health(request: Request):
#     headers = {'Authorization': request.headers.get('Authorization'), 'Content-Type': 'application/json'}
    
#     if not headers['Authorization']:
#         raise HTTPException(status_code=400, detail="Authorization header is missing")
    
    
#     async with aiohttp.ClientSession() as session:
#         async with session.post(RUNPOD_URL +"/run", headers=headers, json=convert_input({"123":"123"}, "/v1/models")) as post_response:
#             if post_response.status != 200:
#                 raise HTTPException(status_code=post_response.status, detail="Failed to initialize the request")
#             job_id = (await post_response.json()).get("id")
#             print(job_id)
#             check_url = f'{RUNPOD_URL}/status/{job_id}'
#             while True:
#                 async with session.post(check_url, headers=headers) as response:
#                     result = await response.json()
#                     if result.get('status') == 'COMPLETED':
#                         # Return healthy status
#                         return {"status": "ok"}
#                 await asyncio.sleep(0.01)

@app.get("/v1/models")
async def get_model(request: Request):
    headers = {'Authorization': request.headers.get('Authorization'), 'Content-Type': 'application/json'}
    
    if not headers['Authorization']:
        raise HTTPException(status_code=400, detail="Authorization header is missing")
    
    
    async with aiohttp.ClientSession() as session:
        async with session.post(RUNPOD_URL +"/run", headers=headers, json=convert_input({"123":"123"}, "/v1/models")) as post_response:
            if post_response.status != 200:
                raise HTTPException(status_code=post_response.status, detail="Failed to initialize the request")
            job_id = (await post_response.json()).get("id")
            check_url = f'{RUNPOD_URL}/status/{job_id}'
            while True:
                async with session.post(check_url, headers=headers) as response:
                    result = await response.json()
                    if result.get('status') == 'COMPLETED':
                        return result.get('output')[0]
                await asyncio.sleep(0.01)
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("middleware:app", host="0.0.0.0", port=8888)