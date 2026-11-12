// DICOM UI checks with a simulated browser. No conversion or inference.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(path.join(__dirname,'..','segmentation_ui.html'),'utf8');
assert(html.includes('<option value="dicom">GRE DICOM series folder</option>'));
const nodes={};
function node(id){
 if(nodes[id])return nodes[id];
 const element={value:id==='inputType'?'nifti':'',textContent:'',children:[],disabled:false,
  checked:true,hidden:false,replaceChildren(){this.children=[];},append(child){this.children.push(child);},
  close(){},showModal(){},focus(){},setCustomValidity(){}};
 if(id==='device'){
  element.value='cpu';element.options=[{value:'auto'},{value:'cpu'}];
 }
 return nodes[id]=element;
}
const requests=[];
const context=vm.createContext({document:{getElementById:node,createElement:()=>({})},
 sessionStorage:{getItem:()=>'',setItem(){}},location:{search:'',pathname:'/'},history:{replaceState(){}},
 URLSearchParams,setTimeout(){},setInterval(){},fetch:(url,options)=>{
  requests.push({url,options});
  if(url.startsWith('/api/browse?')){
   const args=new URL(url,'http://localhost').searchParams;
   const folder=args.get('path');
   return Promise.resolve({ok:true,json:async()=>({path:folder,parent:'/data',entries:[]})});
  }
  return new Promise(()=>{});
 }});
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
(async()=>{
 node('input').value='/data/old.nii.gz';node('inputType').value='dicom';node('inputType').onchange();
 assert.equal(node('input').value,'');
 assert.equal(node('browseInput').textContent,'Browse series folder');
 assert(node('inputHint').textContent.includes('MRIForge'));
 vm.runInContext("startDir='/data';",context);
 await context.openBrowser('dicom');
 assert.equal(node('selectFolder').hidden,false);
 assert.equal(node('selectFolder').textContent,'Use this DICOM series');
 assert(requests.some(r=>r.url.includes('mode=dicom')));
 await context.browse('/data/GRE series');node('selectFolder').onclick();
 assert.equal(node('input').value,'/data/GRE series');
 assert.equal(node('outputParent').value,'/data');
 assert.equal(node('outputName').value,'GRE series_results');
 // DICOM folder selection preserves manually chosen output destinations.
 node('outputParent').value='/results';node('outputName').value='FU_03';
 await context.browse('/data/another GRE');node('selectFolder').onclick();
 assert.equal(node('input').value,'/data/another GRE');assert.equal(node('outputParent').value,'/results');
 node('radius').value='1';node('runForm').onsubmit({preventDefault(){}});
 const body=JSON.parse(requests.filter(r=>r.url==='/api/run').at(-1).options.body);
 assert.equal(body.input_type,'dicom');assert.equal(body.input,'/data/another GRE');
 assert.equal(body.output_name,'FU_03');
 assert.deepEqual(Array.from(context.progressNames({settings:{input_type:'dicom',all_echoes:true}})),
  ['DICOM conversion','Cartilage','Joint model','Meniscus correction','Merge','All echoes']);
 assert.deepEqual(Array.from(context.progressNames({settings:{input_type:'dicom',all_echoes:false}})),
  ['DICOM conversion','Cartilage','Joint model','Meniscus correction','Merge']);
 assert.equal(context.progressNames({settings:{input_type:'nifti',all_echoes:false}}).length,4);
 // Reload restores the original DICOM directory and format, not the generated NIfTI.
 node('input').value='';vm.runInContext('restoreSettings=true;',context);
 context.render({state:'completed',stage:6,total:6,message:'Completed',logs:[],outputs:[],
  python:'/python',start_dir:'/data',settings:{input_type:'dicom',input:'/data/GRE series',
   output:'/results/FU_03',output_parent:'/results',output_name:'FU_03',device:'cpu',radius:1,all_echoes:true}});
 assert.equal(node('inputType').value,'dicom');assert.equal(node('input').value,'/data/GRE series');
 assert.equal(node('steps').children[0].textContent,'1. DICOM conversion');
 node('inputType').value='nifti';node('inputType').onchange();
 context.render({state:'completed',stage:6,total:6,message:'Completed',logs:[],outputs:[],python:'/python',start_dir:'/data',settings:{input_type:'dicom',input:'/data/old GRE',output:'/old',all_echoes:true}});
 assert.equal(node('inputType').value,'nifti');assert.equal(node('input').value,'');
 await context.openBrowser('file');assert.equal(node('selectFolder').hidden,true);
 assert.equal(node('browseInput').textContent,'Browse files');
 console.log('PASS: DICOM/NIfTI selection, series-folder browser, output paths, request, progress and restored settings.');
})().catch(error=>{console.error(error);process.exitCode=1;});
