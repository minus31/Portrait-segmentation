files_str=$(git diff --name-only HEAD HEAD~1)
commit_message=$(git show -s --format=%B HEAD)

files=$(echo $files_str | tr " " "\n")

for file in $files
do
    cp -r $file ../seerslab_bitbucket/seersegmentation/
done

cd ../seerslab_bitbucket/seersegmentation/

git add -A .
git commit -m "$commit_message"
git push origin master

