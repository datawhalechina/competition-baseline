def train_feature(id, path, part):
    df = pd.read_hdf(path + str(id) + '.hdf', part)

    apidict = apidict2.copy()
    apidict['file_id'] = id

    # api 序列合并
    apidict['api'] = ' '.join(df['api'])

    # api 个数统计 词袋 CountVectorizer
    for rows in df['api'].value_counts().reset_index().iterrows():
        apidict[rows[1]['index']] = rows[1]['api']

    # api 重复次数比例
    apidict['api_dpulicate_single'] = sum(df['api'].value_counts().reset_index()['api'] - 1) / df.shape[0]
    # api 连续重复比例
    apidict['api_dpulicate_2ngram'] = sum(df['api'].iloc[:-1].values == df['api'].iloc[1:].values) / df.shape[0]

    # index 重复统计
    apidict['index_dpulicate_flag'] = df.groupby(['tid', 'index'])['api'].nunique().max()
    apidict['index_dpulicate_radio'] = sum(df.groupby(['tid', 'index'])['api'].nunique() - 1) / df.shape[0]

    # api 整体统计
    apidict['api_count'] = df['api'].nunique()
    apidict['api_count_maxratio'] = df['api'].value_counts()[0] / df.shape[0]

    # tid 整体统计
    apidict['tid_count'] = df['tid'].nunique()
    apidict['tid_max_length'] = df.groupby('tid')['index'].count().max()

    # return_value 统计
    apidict['return_value_count'] = df['return_value'].nunique()
    apidict['return_value=0'] = sum(df['return_value'] == 0) / df.shape[0]
    apidict['return_value!=0'] = sum(df['return_value'] != 0) / df.shape[0]
    apidict['return_value==1'] = sum(df['return_value'] == 1) / df.shape[0]
    apidict['return_value=-1'] = sum(df['return_value'] == -1) / df.shape[0]

    apidict['tid_first_value'] = df.groupby('tid').first()['return_value'].mean()
    apidict['tid_first_value!=0'] = sum(df.groupby('tid').first()['return_value'] != 0)
    apidict['tid_last_value'] = df.groupby('tid').last()['return_value'].mean()
    apidict['tid_last_value!=0'] = sum(df.groupby('tid').last()['return_value'] != 0)

    # Behaviour: File, Process, Memory, Register, Network, Service, Other.

    # 注册表信息，注册表修改信息
    reg_cols = ['RegOpenKeyExW', 'RegQueryValueExW', 'RegCloseKey', 'RegOpenKeyExA', 'RegQueryValueExA',
               'RegEnumKeyExW', 'RegQueryInfoKeyW', 'RegEnumValueW', 'RegEnumKeyW', 'RegCreateKeyExW',
               'RegSetValueExW', 'RegEnumValueA', 'RegDeleteValueW', 'RegCreateKeyExA', 'RegEnumKeyExA',
               'RegSetValueExA', 'RegDeleteValueA', 'RegDeleteKeyW', 'RegQueryInfoKeyA', 'RegDeleteKeyA']

    regalter_cols = ['RegCreateKeyExW', 'RegSetValueExW', 'RegDeleteValueW', 'RegCreateKeyExA',
               'RegSetValueExA', 'RegDeleteValueA', 'RegDeleteKeyW', 'RegDeleteKeyA']

    apidict['reg_info'] = int(df[df['api'].isin(reg_cols)].shape[0] > 1)
    apidict['reg_info_ratio'] = df[df['api'].isin(reg_cols)].shape[0] / df.shape[0]

    apidict['reg_infoalter'] = int(df[df['api'].isin(regalter_cols)].shape[0] > 1)
    apidict['reg_infoalter_ratio'] = df[df['api'].isin(regalter_cols)].shape[0] / df.shape[0]
    apidict['reg_infoalter_ratio2'] = df[df['api'].isin(regalter_cols)].shape[0] / (df[df['api'].isin(reg_cols)].shape[0]+1)

    # 网络信息
    network_cols = ['InternetCrackUrlA', 'InternetSetOptionA', 'InternetGetConnectedState', 'InternetOpenW',
                   'InternetSetStatusCallback', 'InternetConnectW', 'InternetQueryOptionA', 'InternetCloseHandle',
                   'InternetOpenA', 'InternetConnectA', 'InternetOpenUrlA', 'InternetReadFile',
                   'InternetGetConnectedStateExW', 'InternetGetConnectedStateExA', 'InternetWriteFile']
    apidict['network_info'] = int(df[df['api'].isin(network_cols)].shape[0] > 1)
    apidict['network_ratio'] = df[df['api'].isin(network_cols)].shape[0] / df.shape[0]

    # 内存信息
    memory_cols = ['NtAllocateVirtualMemory', 'NtFreeVirtualMemory', 'NtProtectVirtualMemory', 'WriteProcessMemory',
                  'ReadProcessMemory', 'NtReadVirtualMemory', 'CryptProtectMemory', 'CryptUnprotectMemory', 'NtWriteVirtualMemory']
    apidict['memory_info'] = int(df[df['api'].isin(memory_cols)].shape[0] > 1)
    apidict['memory_ratio'] = df[df['api'].isin(memory_cols)].shape[0] / df.shape[0]

    # 文件信息
    file_cols = ['NtCreateFile', 'NtWriteFile', 'NtQueryAttributesFile', 'GetFileVersionInfoSizeW', 'GetFileVersionInfoW',
                'NtSetInformationFile', 'NtDeviceIoControlFile', 'NtOpenFile', 'FindFirstFileExW', 'GetFileAttributesW',
                'DeleteFileW', 'CopyFileA', 'SetFilePointer', 'NtReadFile', 'GetFileType', 'SetFileTime',
                'CopyFileW', 'MoveFileWithProgressW', 'CopyFileExW', 'NtDeleteFile']
    filealter_cols = ['NtCreateFile', 'NtWriteFile',
                'DeleteFileW', 'CopyFileA', 'SetFilePointer', 'SetFileTime',
                'CopyFileW', 'MoveFileWithProgressW', 'CopyFileExW', 'NtDeleteFile']

    apidict['file_info'] = int(df[df['api'].isin(file_cols)].shape[0] > 1)
    apidict['file_info_ratio'] = df[df['api'].isin(file_cols)].shape[0] / df.shape[0]

    apidict['file_alter_info'] = int(df[df['api'].isin(filealter_cols)].shape[0] > 1)
    apidict['file_alter_info_ratio'] = df[df['api'].isin(filealter_cols)].shape[0] / df.shape[0]
    apidict['file_alter_info_ratio2'] = apidict['file_alter_info_ratio'] / (apidict['file_info_ratio'] + 0.01)

    # 进程信息
    thread_cols = ['CreateThread', 'Thread32First', 'Thread32Next', 'NtResumeThread', 'NtCreateThreadEx',
                   'NtOpenThread', 'NtTerminateThread', 'NtSuspendThread', 'NtGetContextThread'
                   'CreateRemoteThread', 'NtQueueApcThread', 'RtlCreateUserThread', 'NtSetContextThread',
                   'CreateRemoteThreadEx', 'NtCreateThread']
    apidict['thread_info'] = int(df[df['api'].isin(thread_cols)].shape[0] > 1)
    apidict['thread_ratio'] = df[df['api'].isin(thread_cols)].shape[0] / df.shape[0]
    apidict['thread_last10_ratio'] = df['api'].isin(thread_cols).iloc[-10:].sum() / 10
    apidict['Thread32Next_ratio'] = df[df['api'].isin(['Thread32Next'])].shape[0] / df.shape[0]

    # 服务信息
    # TODO: 成功创建服务的返回值?
    service_cols = ['OpenServiceA', 'CreateServiceA', 'StartServiceA', 'CreateServiceW', 'StartServiceW',
                    'ControlService', 'DeleteService']
    apidict['service_info'] = int(df[df['api'].isin(reg_cols)].shape[0] > 1)
    apidict['service_ratio'] = df[df['api'].isin(service_cols)].shape[0] / df.shape[0]

    # DLL信息
    dll_cols = ['LdrLoadDll', 'LdrUnloadDll', 'LdrGetDllHandle']
    apidict['dll_info'] = int(df[df['api'].isin(dll_cols)].shape[0] > 1)
    apidict['dll_ratio'] = df[df['api'].isin(dll_cols)].shape[0] / df.shape[0]

    # 加密信息
    crypt_cols = ['CryptAcquireContextW', 'CryptProtectMemory', 'CryptUnprotectMemory', 'CryptHashData',
                 'CryptAcquireContextA', 'CryptEncrypt', 'CryptExportKey', 'CryptCreateHash', 'CryptDecodeObjectEx',
                 'CryptProtectData', 'CryptDecrypt', 'CryptUnprotectData']
    apidict['crypt_info'] = int(df[df['api'].isin(crypt_cols)].shape[0] > 1)

    # 证书信息
    cert_cols = ['CertCreateCertificateContext', 'CertOpenSystemStoreA', 'CertOpenSystemStoreW', 'CertOpenStore',
                'CertControlStore']
    apidict['cert_info'] = int(df[df['api'].isin(cert_cols)].shape[0] > 1)

    # COM信息
    com_cols = ['CoCreateInstance', 'CoCreateInstanceEx', 'CoGetClassObject', 'CoInitializeEx', 'CoInitializeSecurity',
               'CoUninitialize', 'ControlService']
    apidict['com_info'] = int(df[df['api'].isin(com_cols)].shape[0] > 1)

    # Find信息
    find_cols = ['FindResourceExW', 'FindResourceA', 'FindFirstFileExW', 'FindWindowA', 'FindResourceW',
                'FindWindowW', 'FindResourceExA', 'FindWindowExW', 'FindFirstFileExA', 'FindWindowExA']
    apidict['find_info'] = int(df[df['api'].isin(find_cols)].shape[0] > 1)

    # Console 信息
    console_cols = ['WriteConsoleA', 'WriteConsoleW']
    apidict['console_cols'] = int(df[df['api'].isin(console_cols)].shape[0] > 1)

    # Control 信息
    control_cols = ['NtDeviceIoControlFile', 'DeviceIoControl', 'ControlService', 'CertControlStore']
    apidict['control_cols'] = int(df[df['api'].isin(control_cols)].shape[0] > 1)

    # Socket 信息
    socket_cols = ['socket', 'setsockopt', 'closesocket', 'getsockname', 'WSASocketW', 'WSASocketA', 'ioctlsocket']
    apidict['socket_cols'] = int(df[df['api'].isin(socket_cols)].shape[0] > 1)

    # Ldr 信息
    ldr_cols  = ['LdrLoadDll', 'LdrGetProcedureAddress', 'LdrUnloadDll', 'LdrGetDllHandle']
    apidict['ldr_info'] = int(df[df['api'].isin(ldr_cols)].shape[0] > 1)
    apidict['ldr_ratio'] = df['api'].isin(ldr_cols).iloc[-10:].sum() / 10

    # Resource 信息
    res_cols = ['FindResourceExW', 'LoadResource', 'FindResourceA', 'SizeofResource', 'FindResourceExA']
    apidict['resource_info'] = int(df[df['api'].isin(res_cols)].shape[0] > 1)

    # Hook 信息
    hook_cols = ['SetWindowsHookExA', 'SetWindowsHookExW', 'UnhookWindowsHookEx']
    apidict['hook_info'] = int(df[df['api'].isin(hook_cols)].shape[0] > 1)

    # Information 信息
    information_cols = ['NtSetInformationFile', 'NtQuerySystemInformation', 'GetTimeZoneInformation',
                       'NtQueryInformationFile', 'GetFileInformationByHandleEx', 'GetFileInformationByHandle',
                       'SetInformationJobObject', 'SetFileInformationByHandle', 'NetGetJoinInformation']
    apidict['information_info'] = int(df[df['api'].isin(information_cols)].shape[0] > 1)

    # Nt 信息
    # TODO

    # Attributes 信息
    attr_cols = ['NtQueryAttributesFile', 'GetFileAttributesW', 'NtQueryFullAttributesFile',
                'SetFileAttributesW']
    apidict['attr_info'] = int(df[df['api'].isin(attr_cols)].shape[0] > 1)

    # Buffer 信息
    buffer_cols = ['RtlCompressBuffer', 'RtlDecompressBuffer']
    apidict['buffer_info'] = int(df[df['api'].isin(buffer_cols)].shape[0] > 1)

    # Module 信息
    module_cols = ['Module32FirstW', 'Module32NextW']
    apidict['module_info'] = int(df[df['api'].isin(module_cols)].shape[0] > 1)

    # reg_info network_info memory_info file_info thread_info service_info find_info ldr_info resource_info information_info
    apidict['type_info1'] = apidict['reg_info'] * apidict['network_info'] * apidict['memory_info']
    apidict['type_info2'] = apidict['network_info'] * apidict['service_info'] * apidict['reg_info']
    apidict['type_info2'] = apidict['ldr_info'] * apidict['file_info'] * apidict['reg_info']
    apidict['type_info3'] = apidict['ldr_info'] * apidict['thread_info'] * apidict['memory_info']
    apidict['type_info4'] = apidict['crypt_info'] * apidict['find_info'] * apidict['network_info']
    apidict['type_info5'] = apidict['resource_info'] * apidict['information_info'] * apidict['network_info']
    apidict['type_info6'] = apidict['crypt_info'] * apidict['memory_info'] * apidict['network_info']

    return apidict

def train_feature0816(id, path, part):
    df = pd.read_hdf(path + str(id) + '.hdf', part)

    apidict = apidict2.copy()
    apidict['file_id'] = id

    # 返回值分析：api调用成功与否？
    #   是否为零、取值空间

    # api的含义、目的，得到新的组合特征

    # 病毒序列分析
