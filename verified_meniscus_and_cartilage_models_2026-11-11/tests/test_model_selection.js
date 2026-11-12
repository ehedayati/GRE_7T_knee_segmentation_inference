// Model-selection UI regression checks. Simulated DOM; no network inference.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(path.join(__dirname,'..','segmentation_ui.html'),'utf8');
const nodes={};
function node(id){
 if(nodes[id])return nodes[id];
 const element={value:'',textContent:'',children:[],disabled:false,checked:true,hidden:false,
  replaceChildren(){this.children=[];},append(child){this.children.push(child);},
  close(){},showModal(){},focus(){},setCustomValidity(){}};
 if(['cartilageModel','meniscusModel','device'].includes(id)){
  element.selected=id==='device'?'auto':'';
  element.children=[{value:element.selected}];
  Object.defineProperty(element,'options',{get(){return this.children;}});
  Object.defineProperty(element,'value',{get(){return this.selected;},set(value){this.selected=this.children.some(x=>x.value===value)?value:'';}});
 }
 return nodes[id]=element;
}
const requests=[];
const context=vm.createContext({document:{getElementById:node,createElement:()=>({})},
 sessionStorage:{getItem:()=>'',setItem(){}},location:{search:'',pathname:'/'},history:{replaceState(){}},
 URLSearchParams,setTimeout(){},setInterval(){},fetch:(url,options)=>{requests.push({url,options});return new Promise(()=>{});}});
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
const cart='/models/cart/top3.pt',alt='/models/cart/top1.pt',men='/models/men/five.pt',menAlt='/models/men/three.pt';
const model=(file,name,def)=>({path:file,name,network:'UNet2',echoes:[0,2,4,6],default:def});
const catalog={defaults:{cartilage:cart,meniscus:men},cartilage:[model(cart,'top3',true),model(alt,'top1',false)],meniscus:[model(men,'five-loss',true),model(menAlt,'three-loss',false)],errors:[]};
context.renderModels(catalog);
assert.equal(node('cartilageModel').value,cart);assert.equal(node('meniscusModel').value,men);
node('cartilageModel').value=alt;node('cartilageModel').onchange();
assert.equal(context.selectedModelPath('cartilage'),alt);
assert.equal(context.selectedModelPath('meniscus'),men);
context.renderModels(catalog);assert.equal(node('cartilageModel').value,alt);
node('input').value='/scan.nii.gz';node('outputParent').value='/results';node('outputName').value='scan';node('radius').value='1';
node('runForm').onsubmit({preventDefault(){}});
const body=JSON.parse(requests.find(r=>r.url==='/api/run').options.body);
assert.equal(body.cartilage_model,alt);assert.equal(body.meniscus_model,men);
// Custom browsing changes only its target network.
node('meniscusModel').value='__custom__';node('meniscusModel').onchange();
assert.equal(node('meniscusCustom').hidden,false);assert.equal(node('meniscusModelPath').required,true);
vm.runInContext("browseMode='checkpoint';modelBrowserKind='meniscus';",context);
context.selectedBrowserFile('/external/new_men.pt');
assert.equal(context.selectedModelPath('meniscus'),'/external/new_men.pt');
assert.equal(context.selectedModelPath('cartilage'),alt);
context.renderModels(catalog);assert.equal(context.selectedModelPath('meniscus'),'/external/new_men.pt');
assert.equal(node('meniscusModel').value,'__custom__');
// Restored settings received before catalog loading must survive the catalog response.
vm.runInContext('modelData=null;',context);
context.restoreModelSelection('cartilage',cart);context.restoreModelSelection('meniscus',menAlt);
context.renderModels(catalog);
assert.equal(node('cartilageModel').value,cart);assert.equal(node('meniscusModel').value,menAlt);
console.log('PASS: independent default/alternative choices, custom checkpoint browsing, refresh preservation, restored settings and submitted model paths.');
