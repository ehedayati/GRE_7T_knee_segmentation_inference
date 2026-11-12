// UI regression checks with a simulated DOM. No GPU queries or inference.
// Run: node tests/test_device_dropdown.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync(path.join(__dirname, '..', 'segmentation_ui.html'), 'utf8');
assert(html.includes('<select id="device"'));
assert(!html.includes('<datalist id="devices"'));
const source = html.split('<script>')[1].split('</script>')[0];
const nodes = {};
function node(id) {
    if (nodes[id]) return nodes[id];
    const element = {value:'', textContent:'', children:[], disabled:false, checked:true,
        setCustomValidity(){}, close(){}, showModal(){}, focus(){},
        replaceChildren(){this.children=[];}, append(child){this.children.push(child);}};
    if (id === 'device') {
        element.children = [{value:'auto',disabled:false}, {value:'cpu',disabled:false}];
        element.selected = 'auto';
        Object.defineProperty(element, 'options', {get(){return this.children;}});
        Object.defineProperty(element, 'value', {
            get(){return this.selected;},
            set(value){this.selected=this.children.some(option=>option.value===value)?value:'';}
        });
    }
    nodes[id] = element;
    return element;
}
const requests=[];
const context = vm.createContext({
    document:{getElementById:node, createElement:()=>({})},
    sessionStorage:{getItem:()=>'', setItem(){}},
    location:{search:'',pathname:'/'}, history:{replaceState(){}},
    URLSearchParams, setInterval(){}, setTimeout(){},
    fetch:(url,options)=>{requests.push({url,options});return new Promise(()=>{});}
});
vm.runInContext(source,context);
const gpu = (index, free=8, error=null) => ({device:'cuda:'+index, name:'GPU '+index,
    free_bytes:free*1024**3,total_bytes:24*1024**3,utilization_percent:null,error});
const snapshot = devices => ({devices, error:null, cuda_available:devices.length>0,
    cuda_visible_devices:'5,2',checked_at:'now'});
const choices = () => node('device').options.map(option=>option.value);
context.renderGPUs(snapshot([gpu(0),gpu(1,18)]));
assert.deepEqual(choices(),['auto','cpu','cuda:0','cuda:1']);
assert(node('device').options[3].textContent.includes('18.0 GiB free'));
assert.equal(node('device').value,'auto');
// Selection must survive a refresh, including a changed availability ordering.
node('device').value='cuda:1';node('device').onchange();
context.renderGPUs(snapshot([gpu(1,12),gpu(0)]));
assert.equal(node('device').value,'cuda:1');
// The selected value, rather than its display label, must be submitted.
node('input').value='/scan.nii.gz';node('outputParent').value='/results';node('outputName').value='FU_03';
node('radius').value='1.0';
node('runForm').onsubmit({preventDefault(){}});
const request=requests.find(item=>item.url==='/api/run');
assert.equal(JSON.parse(request.options.body).device,'cuda:1');
// CPU survives refresh, and invalid/removed GPUs fall back to auto.
node('device').value='cpu';node('device').onchange();
context.renderGPUs(snapshot([gpu(0)]));assert.equal(node('device').value,'cpu');
node('device').value='cuda:0';node('device').onchange();
context.renderGPUs(snapshot([]));assert.equal(node('device').value,'auto');
assert.deepEqual(choices(),['auto','cpu']);
context.renderGPUs(snapshot([gpu(0,0,'unreadable'),gpu(1)]));
assert(node('device').options.find(option=>option.value==='cuda:0').disabled);
// A failed check does not discard the existing dropdown options/selection.
node('device').value='cuda:1';node('device').onchange();
context.renderGPUs({devices:[],error:'query failed'});
assert.equal(node('device').value,'cuda:1');
assert(choices().includes('cuda:1'));
// Restoring a run before the first GPU response must retain the requested choice.
vm.runInContext("gpuDevicesLoaded=false;",context);
context.populateDeviceOptions([]);
vm.runInContext("gpuDevicesLoaded=false;",context);
context.restoreDeviceSelection('cuda:1');
context.renderGPUs(snapshot([gpu(0),gpu(1)]));
assert.equal(node('device').value,'cuda:1');
console.log('PASS: native dropdown, discovered GPUs, selection persistence, restored selection, submitted cuda:N, disabled GPUs, CPU and query-error handling.');

// Optional utilization dependencies must not invalidate a readable GPU.
const missingNvml='nvidia-ml-py is not available. Install: python -m pip install nvidia-ml-py';
const optional=snapshot([gpu(0)]);
optional.warnings=['GPU utilization is unavailable: '+missingNvml];
optional.missing_packages=['nvidia-ml-py'];
optional.devices[0].utilization_error=missingNvml;
context.renderGPUs(optional);
assert(node('gpuSummary').textContent.includes(missingNvml));
assert(!node('gpuSummary').textContent.includes('GPU check failed'));
assert(node('gpuList').children[0].textContent.includes(missingNvml));
assert(!node('device').options.find(option=>option.value==='cuda:0').disabled);
// Dependency failures show a specific message; unrelated failures retain theirs.
context.renderGPUs({devices:[],error:missingNvml,missing_packages:['nvidia-ml-py']});
assert.equal(node('gpuSummary').textContent,missingNvml);
context.renderGPUs({devices:[],error:'CUDA driver failure',missing_packages:[]});
assert.equal(node('gpuSummary').textContent,'GPU check failed: CUDA driver failure');
console.log('PASS: missing nvidia-ml-py diagnostics, usable GPU memory, distinct driver errors.');
