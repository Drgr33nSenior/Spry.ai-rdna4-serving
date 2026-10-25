"""Explicit pinned BetterBench client patch for native terminal protocol records."""
import hashlib

ORIGINAL = 'b8fbd0a632e05673f78882a902c191bd6ae11af0e9bc544b6f0a08b8939b7bfd'


def one(text, before, after):
    assert text.count(before) == 1, (before, text.count(before))
    return text.replace(before, after)


def derive(original):
    assert hashlib.sha256(original).hexdigest() == ORIGINAL
    text = original.decode()
    text = one(text, 'import http.client\n', 'import http.client\nimport hashlib\n')
    text = one(text, '    error: str | None = None\n', '''    error: str | None = None
    native_response_id: str | None = None
    native_speculation: dict | None = None
    native_protocol: dict = field(default_factory=dict)
''')
    text = one(text, '    t0 = time.perf_counter()\n    try:\n', '''    t0 = time.perf_counter()
    protocol = res.native_protocol
    protocol.update(schema=1,data_frames=0,metadata_frames=0,usage_frames=0,
        finish_markers=0,done_markers=0,metadata_index=None,usage_index=None,
        finish_index=None,done_index=None,finish_frame=None,finish_frame_sha256=None,
        usage_frame=None,usage_frame_sha256=None,done_frame=None,done_frame_sha256=None,complete=False)
    native_done = False
    def unique_pairs(rows):
        result = {}
        for key,value in rows:
            if key in result:raise ValueError("duplicate native stream JSON key")
            result[key]=value
        return result
    try:
''')
    text = one(text, '            if data == "[DONE]":\n                break\n', '''            protocol["data_frames"] += 1
            if native_done:
                raise ValueError("data after native terminal completion")
            if data == "[DONE]":
                if not (protocol["metadata_frames"] == protocol["usage_frames"] == protocol["finish_markers"] == 1):
                    raise ValueError("incomplete native terminal markers")
                native_done = True
                protocol["done_markers"] += 1
                protocol["done_index"] = protocol["data_frames"]
                protocol["done_frame"] = data
                protocol["done_frame_sha256"] = hashlib.sha256(data.encode("utf-8")).hexdigest()
                continue
''')
    text = one(text, '''            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
''', '''            try:
                obj = json.loads(data,object_pairs_hook=unique_pairs,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite stream JSON")))
            except (json.JSONDecodeError,ValueError) as error:
                raise ValueError("malformed native stream JSON") from error
            if type(obj) is not dict:
                raise ValueError("native stream JSON object required")
            if type(obj.get("choices")) is not list:
                raise ValueError("native stream choice array required")
''')
    text = one(text, '            if obj.get("usage"):\n                usage = obj["usage"]\n', '''            response_id = obj.get("id")
            if type(response_id) is not str or not 0 < len(response_id) <= 128:
                raise ValueError("native stream response identity missing")
            if res.native_response_id is None:
                res.native_response_id = response_id
            elif res.native_response_id != response_id:
                raise ValueError("conflicting native stream response identity")
            if obj.get("model") != model:
                raise ValueError("native stream model identity differs")
            extension = obj.get("sglext")
            if extension is not None:
                if type(extension) is not dict:
                    raise ValueError("malformed native stream extension")
                details = extension.get("spec_tokens_details")
                if details is not None:
                    if res.native_speculation is not None:
                        raise ValueError("duplicate native speculation extension")
                    if obj.get("choices") != [] or type(details) is not dict:
                        raise ValueError("native speculation must be a separate terminal extension")
                    if protocol["finish_markers"] != 1 or protocol["usage_frames"]:
                        raise ValueError("native speculation extension order")
                    raw_frame = data.encode("utf-8")
                    if len(raw_frame) > 16384:
                        raise ValueError("native speculation extension bound")
                    res.native_speculation = dict(response_id=response_id,
                        raw_frame=data,raw_frame_sha256=hashlib.sha256(raw_frame).hexdigest(),details=details)
                    protocol["metadata_frames"] += 1
                    protocol["metadata_index"] = protocol["data_frames"]
            if obj.get("usage") is not None:
                if type(obj["usage"]) is not dict or protocol["usage_frames"] or res.native_speculation is None or obj.get("choices") != []:
                    raise ValueError("duplicate, malformed or early native usage")
                protocol["usage_frames"] += 1
                protocol["usage_index"] = protocol["data_frames"]
                protocol["usage_frame"] = data
                protocol["usage_frame_sha256"] = hashlib.sha256(data.encode("utf-8")).hexdigest()
                if len(data.encode("utf-8")) > 16384:raise ValueError("native usage frame bound")
                usage = obj["usage"]
''')
    text = one(text, '''            if not choices:
                continue
            ch = choices[0]
            if ch.get("finish_reason"):
                res.finish_reason = ch["finish_reason"]
''', '''            if not choices:
                continue
            if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
                raise ValueError("one native stream choice required")
            if protocol["finish_markers"] or res.native_speculation is not None:
                raise ValueError("conflicting native completion after finish")
            ch = choices[0]
            if ch.get("finish_reason") is not None:
                if type(ch["finish_reason"]) is not str or not ch["finish_reason"]:
                    raise ValueError("invalid native finish reason")
                res.finish_reason = ch["finish_reason"]
                protocol["finish_markers"] += 1
                protocol["finish_index"] = protocol["data_frames"]
                protocol["finish_frame"] = data
                protocol["finish_frame_sha256"] = hashlib.sha256(data.encode("utf-8")).hexdigest()
                if len(data.encode("utf-8")) > 16384:raise ValueError("native finish frame bound")
''')
    text = one(text, '    return _finalize(res, tl, t0, time.perf_counter(), usage)\n', '''    if not native_done or res.native_speculation is None:
        res.error = "incomplete native speculation terminal stream"
        return res
    protocol["complete"] = True
    return _finalize(res, tl, t0, time.perf_counter(), usage)
''')
    compile(text, '<betterbench-native-stream-protocol-client>', 'exec', dont_inherit=True)
    return text.encode()
